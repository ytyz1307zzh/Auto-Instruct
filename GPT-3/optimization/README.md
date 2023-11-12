# Auto-Instruct Model

1. Candidate instruction generation: `instruction_generation/`
2. Label the candidate instructions and get the train/dev/test sets: `instruction_labeling`
3. Pre-process the train/dev/test sets for pytorch training: `build_dataset.py`
```bash
python GPT-3/optimization/build_dataset.py \
    -builder_script GPT-3/optimization/Dataset.py \
    -train_file DATASET_PATH/train.jsonl \
    -valid_file DATASET_PATH/dev.jsonl \
    -test_file DATASET_PATH/test.jsonl \
    -output_dir DATASET_PATH
```
4. Train the model with `train.py`
```bash
wandb login YOUR_WANDB_KEY

python -m torch.distributed.launch \
    --nproc_per_node=8 \
    GPT-3/optimization/run.py \
    --data_dir data/class_niv2_fewshot \
    --output_dir YOUR_SAVE_DIR \
    --do_train \
    --do_test \
    --model_name google/flan-t5-large \
    --train_batch_size 4 \
    --test_batch_size 1 \
    --learning_rate 5e-5 \
    --constant_lr True \
    --weight_decay 0.0 \
    --dropout 0.1 \
    --max_grad_norm 1.0 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpoint True \
    --num_train_epochs 5 \
    --wait_steps 1000 \
    --eval_steps 250 \
    --rolling_loss_steps 25 \
    --output_max_length 1 \
    --seed 42 \
    --wandb_project_name Auto-Instruct \
    --wandb_username YOUR_WANDB_USERNAME \
    --prefix 91test_
```

