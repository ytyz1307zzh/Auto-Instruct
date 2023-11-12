"""
Training the instruction selection model
"""

import os
import pdb
import time
import math
import wandb
import torch
import random
import numpy as np
from tqdm import tqdm
from Logger import logger
from datasets import load_from_disk
from train_utils import (
    calculate_num_steps,
    get_attention_mask,
    get_labels,
    mean_reduce,
    sum_reduce,
    flatten_batch,
    calc_loss,
    evaluate_prediction
)
from data_utils import (
    save_predictions_to_json,
    combine_prediction_files,
    save_checkpoint
)
from transformers import T5TokenizerFast, T5Config, T5ForConditionalGeneration
from transformers.optimization import Adafactor, AdafactorSchedule, AdamW, get_linear_schedule_with_warmup
from Collator import InstructionSelectionCollator, InstructionSelectionDataLoader


def run(args):
    
    # Set of distributed training
    logger.info(f"args.local_rank {args.local_rank}")
    world_size = os.environ.get("WORLD_SIZE")
    args.world_size = int(world_size) if world_size else 1
    logger.info(f"World size: {args.world_size}")

    if args.world_size > 1:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.device_type = "cuda"
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    elif torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.device_type = "cuda"
    else:
        args.device = torch.device("cpu")
        args.device_type = "cpu"

    # Load model and tokenizer from Huggingface
    config = T5Config.from_pretrained(args.model_name)
    if args.dropout is not None:
        config.dropout_rate = args.dropout  # Set the dropout rate to what we input from command line
    tokenizer = T5TokenizerFast.from_pretrained(args.model_name)

    # Load the pre-processed dataset
    dataset = load_from_disk(args.data_dir)
    logger.info(f"Loaded dataset from {args.data_dir}")
    train_set, valid_set, test_set = None, None, None
    if args.do_train:
        train_set = dataset["train"]
        valid_set = dataset["validation"]
    if args.do_test:
        if args.test_on_dev:
            test_set = dataset["validation"]
        elif args.test_on_train:
            test_set = dataset["train"]
        else:
            test_set = dataset["test"]

        if args.sample_test_data is not None:
            test_set = test_set.select(range(args.sample_test_data))

    data_collator = InstructionSelectionCollator(config=config)

    if args.do_train:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name, config=config)

        if args.gradient_checkpoint:
            logger.info("Enabling gradient checkpointing...")
            model.gradient_checkpointing_enable()

        # for multi-gpu single-node training, use DDP
        if args.world_size > 1:
            model.to(args.device)
            logger.warning(f"Moving model to {args.device}...")
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )
        # otherwise, only use one gpu or use cpu
        else:
            model.to(args.device)

        train_steps = calculate_num_steps(args, len(train_set))
        args.total_global_steps = train_steps["total_global_steps"]
        args.total_local_steps = train_steps["total_local_steps"]
        logger.info(f"Total global training steps: {args.total_global_steps}, "
                    f"Total local training steps: {args.total_local_steps}")
        # args.warmup_steps = train_steps["warmup_steps"]
        # logger.info(f"Total global training steps: {args.total_global_steps}, "
        #             f"Total local training steps: {args.total_local_steps}, "
        #             f"Warmup steps: {args.warmup_steps}")

        if args.optimizer == 'adafactor':
            if args.constant_lr:
                optimizer = Adafactor(
                    model.parameters(),
                    lr=args.learning_rate,
                    clip_threshold=args.max_grad_norm,
                    weight_decay=args.weight_decay,
                    scale_parameter=False,
                    relative_step=False,
                    warmup_init=False
                )
            else:
                optimizer = Adafactor(
                    model.parameters(),
                    lr=None,
                    clip_threshold=args.max_grad_norm,
                    weight_decay=args.weight_decay,
                    scale_parameter=True,
                    relative_step=True,
                    warmup_init=True
                )
            scheduler = AdafactorSchedule(optimizer, initial_lr=args.learning_rate)

        elif args.optimizer == 'adamw':
            no_decay = ['bias', 'layer_norm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=round(args.warmup_ratio * args.total_global_steps),
                num_training_steps=args.total_global_steps
            )

        else:
            raise ValueError(f"Invalid optimizer choice {args.optimizer}")

        scaler = torch.cuda.amp.GradScaler() if args.half_precision else None

        train(
            args=args,
            model=model,
            train_set=train_set,
            valid_set=valid_set,
            data_collator=data_collator,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler
        )

    # sync all processes if continue to do prediction
    if args.do_train and args.do_test and args.world_size > 1:
        torch.distributed.barrier()

    if args.do_test:
        if args.world_size > 1:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
        elif torch.cuda.is_available():
            map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}
        else:
            map_location = {'cuda:%d' % 0: 'cpu'}

        model = T5ForConditionalGeneration.from_pretrained(args.model_name)

        # Load checkpoint from output_dir
        state_dict = torch.load(os.path.join(args.output_dir, "pytorch_model.bin"), map_location=map_location)
        model.load_state_dict(state_dict)
        logger.info("Loaded checkpoint from {}".format(args.output_dir))

        # for multi-gpu single-node running, use DDP
        if args.world_size > 1:
            model.to(args.device)
            logger.warning(f"Moving model to {args.device}...")
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )
        # otherwise, only use one gpu or use cpu
        else:
            model.to(args.device)
            
        model.eval()

        test_score, random_test_score, oracle_test_score = predict(
            args=args,
            model=model if args.world_size == 1 else model.module,
            test_set=test_set,
            data_collator=data_collator,
            tokenizer=tokenizer,
            save_model=True
        )

        if args.test_on_dev:
            logger.info(f"Accuracy on validation data: {test_score:.2%}, "
                        f"random selection: {random_test_score:.2%}, "
                        f"oracle selection: {oracle_test_score:.2%}")
        elif args.test_on_train:
            logger.info(f"Accuracy on train data: {test_score:.2%}, "
                        f"random selection: {random_test_score:.2%}, "
                        f"oracle selection: {oracle_test_score:.2%}")
        else:
            logger.info(f"Accuracy on test data: {test_score:.2%}, "
                        f"random selection: {random_test_score:.2%}, "
                        f"oracle selection: {oracle_test_score:.2%}")

        # sync all processes before combining prediction files
        if args.world_size > 1:
            torch.distributed.barrier()

        if args.world_size > 1 and args.local_rank == 0:
            combine_prediction_files(args.output_dir, args.world_size, args.prefix)


def train(
    args,
    model,
    train_set,
    valid_set,
    data_collator,
    tokenizer,
    optimizer,
    scheduler,
    scaler=None
):
    model.train()
    global_step = 0
    local_step = 0
    wait_step = 0
    train_loss_host = 0.0
    best_score = -10000
    stop_training = False

    if args.sample_train_data is not None:
        train_set = train_set.select(range(args.sample_train_data))
        logger.info(f"Down-sample training set to {args.sample_train_data} examples!")

    train_dataloader = InstructionSelectionDataLoader(
        args,
        dataset=train_set,
        collate_fn=data_collator,
        is_training=True
    )

    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

    train_start_time = time.time()
    logger.info('Start training!')

    ignore_fields = ["id"]

    for epoch in range(int(args.num_train_epochs)):
        logger.info('*' * 20 + f"Epoch {epoch + 1}" + "*" * 20)
        for batch in train_dataloader:
            local_step += 1
            batch_size = batch["input_ids"].size(0)
            batch["attention_mask"] = get_attention_mask(batch["input_ids"], pad_id=tokenizer.pad_token_id)
            batch = {key: value.to(args.device) for key, value in batch.items() if key not in ignore_fields}

            # Convert input ids to (batch_size * num_instructions, input_seq_length)
            batch = flatten_batch(batch)

            if args.local_rank != -1 and not (local_step % args.gradient_accumulation_steps == 0 or
                    local_step == args.total_local_steps):
                # If this step is a gradient accumulation step
                # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                with model.no_sync():
                    loss, logits = train_step(args, model, batch, scaler, loss_fn=kl_loss, batch_size=batch_size)
            else:
                loss, logits = train_step(args, model, batch, scaler, loss_fn=kl_loss, batch_size=batch_size)

            # get the average loss value across gpus (just for logging),
            # the distributed backward pass is handled by DDP itself
            loss_value = loss.clone().detach()
            mean_reduce(loss_value, args)
            train_loss_host += loss_value.item()

            # Gradient Accumulation
            # Always make an optimization step at the last batch of training
            if local_step % args.gradient_accumulation_steps == 0 or \
                    local_step == args.total_local_steps:

                # Unscales gradient before gradient clipping
                # if args.half_precision:
                #     scaler.unscale_(optimizer)

                # Gradient Clipping
                if args.optimizer != "adafactor":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if args.half_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()  # We have accumulated enough gradients

                if args.optimizer != "adafactor":
                    scheduler.step()
                model.zero_grad()
                global_step += 1

                if global_step % args.rolling_loss_steps == 0:
                    curr_lr = scheduler.get_lr()
                    rolling_loss = train_loss_host / args.rolling_loss_steps
                    logger.info(f"Training step {global_step}: Loss {rolling_loss}, Current LR: {curr_lr}")

                    if args.half_precision:
                        grad_scale = scaler.get_scale()
                        logger.info(f'Gradient scale for fp16: {grad_scale}')

                    if args.local_rank == -1 or args.local_rank == 0:
                        wandb.log({"train_kl_loss": round(rolling_loss, 4)})

                    train_loss_host = 0  # Clear the recorded loss

                # evaluate on dev set
                # Always evaluate at the last batch of training
                if global_step % args.eval_steps == 0 or \
                        global_step == args.total_global_steps:
                    model.eval()
                    train_duration = time.time() - train_start_time
                    logger.info(f"Train duration: {train_duration:.3f}s")

                    eval_score, random_eval_score, oracle_eval_score = predict(
                        args=args,
                        model=model if args.world_size == 1 else model.module,
                        test_set=valid_set,
                        data_collator=data_collator,
                        tokenizer=tokenizer,
                        save_model=False
                    )

                    # Logging
                    logger.info(f"Evaluation on training step {global_step}, "
                                f"Accuracy {eval_score:.2%}, "
                                f"random selection {random_eval_score:.2%}, "
                                f"oracle selection {oracle_eval_score:.2%}"
                                f"epoch={epoch + 1}")
                    if args.local_rank == -1 or args.local_rank == 0:
                        wandb.log({"eval_accuracy": round(eval_score * 100, 2)})

                    if best_score < eval_score:
                        # while DDP, only save model using the main process
                        if args.world_size > 1 and args.local_rank == 0:
                            save_checkpoint(model.module, args.output_dir)
                        elif args.world_size <= 1:
                            save_checkpoint(model, args.output_dir)
                        # Logging
                        logger.info("Saving model with best accuracy: %.2f%% -> %.2f%% on epoch=%d, global_step=%d" %
                                    (best_score * 100.0, eval_score * 100.0, epoch + 1, global_step))
                        logger.info("Checkpoint saved to {}".format(args.output_dir))

                        best_score = eval_score
                        wait_step = 0
                        stop_training = False

                    else:
                        # If not the best score, still save the model as the "last checkpoint"
                        # while DDP, only save model using the main process
                        if args.world_size > 1 and args.local_rank == 0:
                            save_checkpoint(model.module, os.path.join(args.output_dir, 'last_checkpoint'))
                        elif args.world_size <= 1:
                            save_checkpoint(model, os.path.join(args.output_dir, 'last_checkpoint'))

                        wait_step += 1
                        logger.info(f"Wait steps = {wait_step}")
                        # early stopping if accuracy did not improve for args.wait_step evaluation rounds
                        if wait_step >= args.wait_steps:
                            stop_training = True
                            logger.info(f"Early stopping!")
                            break

                    model.train()
                    train_start_time = time.time()

        if stop_training:
            break

    logger.info("Best accuracy on validation data: %.2f" % (best_score * 100))


def predict(
    args,
    model,
    test_set,
    data_collator,
    tokenizer,
    save_model: bool
):
    """
    Make predictions during inference.
    save_model: if True, then this is the test set (needs to save output); if False, then this is the dev set
    """
    all_predict_positive_scores = []  # (num_examples, num_instructions)
    all_select_instruction_indices = []  # (num_examples,)
    all_instruction_rouge_scores = []  # (num_examples, num_instructions)
    all_select_instruction_rouge_scores = []  # (num_examples,)
    all_random_instruction_rouge_scores = []  # (num_examples,)
    all_oracle_instruction_rouge_scores = []  # (num_examples,)
    all_data_ids = []  # (num_examples,)
    data_cnt = 0  # total number of instances in this node

    test_dataloader = InstructionSelectionDataLoader(
        args,
        dataset=test_set,
        collate_fn=data_collator,
        is_training=False
    )

    eval_start_time = time.time()
    ignore_fields = ["id"]

    for i, batch in enumerate(test_dataloader):
        all_data_ids.extend(batch["id"])  # (batch,)
        batch_size = batch["input_ids"].size(0)
        batch["attention_mask"] = get_attention_mask(batch["input_ids"], pad_id=tokenizer.pad_token_id)
        batch = {key: value.to(args.device) for key, value in batch.items() if key not in ignore_fields}
        batch = flatten_batch(batch)

        if args.half_precision:
            if args.fp16:
                float_dtype = torch.float16
            elif args.bf16:
                float_dtype = torch.bfloat16
            else:
                raise ValueError

            with torch.autocast(device_type=args.device_type, dtype=float_dtype):
                output_scores = generate_predictions(model, batch, args.test_mini_batch_size, args.output_max_length)
        else:
            output_scores = generate_predictions(model, batch, args.test_mini_batch_size, args.output_max_length)

        positive_scores, select_instruction_scores, select_instruction_indices = evaluate_prediction(
            output_scores=output_scores,
            instruction_rouge_scores=batch["instruction_rouge"],
            batch_size=batch_size
        )

        all_predict_positive_scores.extend(positive_scores.tolist())  # (batch, num_instructions)
        all_select_instruction_indices.extend(select_instruction_indices.tolist())  # (batch,)
        all_instruction_rouge_scores.extend(batch["instruction_rouge"].tolist())  # (batch, num_instructions)
        all_select_instruction_rouge_scores.extend(select_instruction_scores.tolist())  # (batch,)
        all_random_instruction_rouge_scores.extend(torch.mean(batch["instruction_rouge"], dim=1).tolist())  # (batch,)
        all_oracle_instruction_rouge_scores.extend(torch.max(batch["instruction_rouge"], dim=1).values.tolist())  # (batch,)

        data_cnt += batch_size

    save_prefix = args.prefix

    if save_model:
        if args.world_size > 1:
            save_path = os.path.join(args.output_dir, "{}predictions_ps{}.json".format(save_prefix, args.local_rank))
        else:
            save_path = os.path.join(args.output_dir, "{}predictions.json".format(save_prefix))

        save_predictions_to_json(
            all_data_ids,
            all_predict_positive_scores,
            all_select_instruction_indices,
            all_instruction_rouge_scores,
            save_path
        )
        logger.info(f"Saved {len(all_data_ids)} predictions to {save_path}")

    # calculate accuracy score
    score = sum(all_select_instruction_rouge_scores) / data_cnt
    random_score = sum(all_random_instruction_rouge_scores) / data_cnt
    oracle_score = sum(all_oracle_instruction_rouge_scores) / data_cnt

    eval_duration = time.time() - eval_start_time
    logger.info(f"Eval duration: {eval_duration:.3f}s")

    # if multi-gpu, we need to compute weighted average accuracy across gpus
    if args.world_size > 1:
        score = torch.tensor(score).to(model.device)
        score = score * data_cnt
        # weighted sum + divide by total number of data objects for average accuracy
        sum_reduce(score, args)

        # The rouge score of randomly selecting instructions
        random_score = torch.tensor(random_score).to(model.device)
        random_score = random_score * data_cnt
        sum_reduce(random_score, args)

        # The rouge score of selecting the best instruction
        oracle_score = torch.tensor(oracle_score).to(model.device)
        oracle_score = oracle_score * data_cnt
        sum_reduce(oracle_score, args)

        data_cnt = torch.tensor(data_cnt).to(model.device)
        sum_reduce(data_cnt, args)  # The total number of instructions in all processes

        score = score / data_cnt
        random_score = random_score / data_cnt
        oracle_score = oracle_score / data_cnt
        # send the final EM to all processes
        # torch.distributed.broadcast(score, src=0)
        return score.item(), random_score.item(), oracle_score.item()
    else:
        return score, random_score, oracle_score


def calc_logits(model, batch, mini_batch_size):
    """
    batch: input_ids, attention_mask and decoder_input_ids are all (batch * num_instructions, seq_length)
    """
    all_logits = []
    batch_size = batch["input_ids"].size(0)
    if mini_batch_size is None or mini_batch_size >= batch_size:
        mini_batch_size = batch_size

    num_mini_batches = math.ceil(batch_size / mini_batch_size)

    for mini_batch_idx in range(num_mini_batches):
        mini_batch = {key: value[mini_batch_idx * mini_batch_size: (mini_batch_idx + 1) * mini_batch_size]
                      for key, value in batch.items()}
        model_outputs = model(
            input_ids=mini_batch["input_ids"],
            attention_mask=mini_batch["attention_mask"],
            decoder_input_ids=mini_batch["decoder_input_ids"],
            use_cache=False,
            return_dict=True
        )
        logits = model_outputs.logits  # (mini_batch, out_seq_length, num_vocab)
        all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=0)
    return all_logits


def train_step(
    args,
    model,
    batch,
    scaler,
    loss_fn,
    batch_size: int
):
    if args.half_precision:
        if args.fp16:
            float_dtype = torch.float16
        elif args.bf16:
            float_dtype = torch.bfloat16
        else:
            raise ValueError

        with torch.autocast(device_type=args.device_type, dtype=float_dtype):
            logits = calc_logits(model, batch, args.train_mini_batch_size)
            loss = calc_loss(
                output_logits=logits,
                gold_labels=batch["instruction_labels"],
                loss=loss_fn,
                batch_size=batch_size
            )
            loss = loss / args.gradient_accumulation_steps
    else:
        logits = calc_logits(model, batch, args.train_mini_batch_size)
        loss = calc_loss(
            output_logits=logits,
            gold_labels=batch["instruction_labels"],
            loss=loss_fn,
            batch_size=batch_size
        )
        loss = loss / args.gradient_accumulation_steps

    # If loss is NaN, directly stop training
    if torch.isnan(loss).data:
        logger.warning("Loss=%s!" % loss.data)
        # logger.error("Stop training because loss=%s" % loss.data)
        # stop_training = True
        # break

    if args.half_precision:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return loss, logits


def generate_predictions(model, batch, mini_batch_size, output_max_length):
    """
    batch: input_ids, attention_mask and decoder_input_ids are all (batch * num_instructions, seq_length)
    """
    all_scores = []
    batch_size = batch["input_ids"].size(0)
    if mini_batch_size is None or mini_batch_size >= batch_size:
        mini_batch_size = batch_size

    num_mini_batches = math.ceil(batch_size / mini_batch_size)

    with torch.inference_mode():
        for mini_batch_idx in range(num_mini_batches):
            mini_batch = {key: value[mini_batch_idx * mini_batch_size: (mini_batch_idx + 1) * mini_batch_size]
                          for key, value in batch.items()}
            model_outputs = model.generate(
                mini_batch["input_ids"],
                max_length=output_max_length + 1,  # taking <pad> into account
                return_dict_in_generate=True,
                output_scores=True
            )
            scores = model_outputs.scores  # a tuple of (mini_batch_size, num_vocab)
            scores = scores[0]  # we only need the score of the first generated token, (mini_batch_size, num_vocab)
            all_scores.append(scores)

    all_scores = torch.cat(all_scores, dim=0)  # (batch * num_instructions, num_vocab)
    return all_scores

