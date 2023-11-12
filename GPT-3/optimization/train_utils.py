import math
import pdb

import torch
from typing import List, Dict, Tuple
import torch.distributed as dist
import torch.nn.functional as F

POS_IDX = 4273
NEG_IDX = 150

POS_CLASS = 1
NEG_CLASS = 0

IDX2LABEL = {
    4273: 1,
    150: 0
}

LABEL2IDX = {
    1: 4273,
    0: 150
}


def calculate_num_steps(args, data_size: int):
    total_train_batch_size = args.train_batch_size
    if args.world_size > 1:
        total_train_batch_size *= args.world_size
        data_size = math.ceil(data_size / args.world_size) * args.world_size
        print("Data size for each epoch under DistributedSampler: ", data_size)
    total_epochs = args.num_train_epochs
    # if gradient_accumulation_steps == 1, then total_local_steps == total_global_steps
    total_local_steps = math.ceil(data_size / total_train_batch_size) * total_epochs
    total_global_steps = math.ceil(total_local_steps / args.gradient_accumulation_steps)
    # warmup_steps = round(total_global_steps * args.warmup_ratio)
    return {
        "total_local_steps": total_local_steps,
        "total_global_steps": total_global_steps,
        # "warmup_steps": warmup_steps
    }


def calc_loss(output_logits: torch.Tensor, gold_labels: torch.Tensor, loss, batch_size: int):
    """
    Calculate KL Divergence of the instruction classification distribution
    :param output_logits: (batch * num_instructions, out_seq_length, num_vocab)
    :param gold_labels: normalized rouge score of each instruction, (batch, num_instructions)
    :param loss: the KLDivLoss object
    :param batch_size: batch size
    """
    assert output_logits.dim() == 3 and gold_labels.dim() == 2
    assert output_logits.size(0) == gold_labels.size(0) * gold_labels.size(1)
    num_instructions = output_logits.size(0) // batch_size

    # Pick the first predicted token logits, then normalize to the distribution of instruction selection
    # The probability of selecting an instruction is positively correlated to the logits of generating "yes"
    output_logits = output_logits[:, 0, :]  # (batch * num_instructions, num_vocab)
    positive_logits = output_logits[:, LABEL2IDX[POS_CLASS]]  # (batch * num_instructions,)
    positive_logits = positive_logits.view(batch_size, num_instructions)
    instruction_logprobs = F.log_softmax(positive_logits, dim=1)  # (batch, num_instructions)

    # Gold labels are already normalized by softmax, so it is already the target distribution
    return loss(instruction_logprobs, gold_labels)


def get_attention_mask(input_ids: torch.LongTensor, pad_id: int = 0):
    attention_mask = torch.where(input_ids == pad_id, 0, 1)
    return attention_mask


def get_labels(output_ids: torch.LongTensor, pad_id: int = 0, replace_id: int = -100):
    labels = torch.where(output_ids == pad_id, replace_id, output_ids)
    return labels


def mean_reduce(x, args):
    if args.world_size > 1:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x.divide_(args.world_size)
    return x


def sum_reduce(x, args):
    if args.world_size > 1:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def flatten_batch(batch):
    """
    Batch: ids, input_ids, output_ids
    Change input_ids from (batch, num_instructions, input_seq_length) to (batch * num_instructions, input_seq_length)
    """
    batch_size, num_instructions, _ = batch["input_ids"].shape
    input_fields = ["input_ids", "attention_mask", "decoder_input_ids"]
    for key in input_fields:
        batch[key] = batch[key].view(batch_size * num_instructions, -1)
    return batch


def evaluate_prediction(output_scores: torch.Tensor, instruction_rouge_scores: torch.Tensor, batch_size: int):
    """
    Rank the instructions according to the score of the positive class
    :param output_scores: the output score of first generated token, (batch * num_instructions, num_vocab)
    :param instruction_rouge_scores: a batch of instruction rouge scores, (batch, num_instructions)
    :param batch_size: batch size
    """
    assert output_scores.dim() == instruction_rouge_scores.dim() == 2
    assert output_scores.size(0) == instruction_rouge_scores.size(0) * instruction_rouge_scores.size(1)
    num_instructions = output_scores.size(0) // batch_size

    positive_scores = output_scores[:, POS_IDX]  # (batch * num_instructions,)
    positive_scores = positive_scores.view(batch_size, num_instructions)
    sorted_scores, sorted_indices = torch.sort(positive_scores, dim=1, descending=True)  # (batch, num_instructions)
    top1_indices = sorted_indices[:, 0]  # (batch,)

    # instruction_rouge_scores: the real rouge score of each instruction output
    # (batch,), the rouge score of top-1 ranked instructions for each example
    select_instruction_scores = instruction_rouge_scores[torch.arange(0, batch_size), top1_indices]
    return positive_scores, select_instruction_scores, top1_indices
