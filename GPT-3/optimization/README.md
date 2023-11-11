# Auto-Instruct Model

1. Candidate instruction generation: `instruction_generation/`
2. Label the candidate instructions and get the train/dev/test sets: `instruction_labeling`
3. Pre-process the train/dev/test sets for pytorch training: `build_dataset.py`
"""bash
python GPT-3/optimization/build_dataset.py \
    -builder_script GPT-3/optimization/Dataset.py \
    -train_file DATASET_PATH/train.jsonl \
    -valid_file DATASET_PATH/dev.jsonl \
    -test_file DATASET_PATH/test.jsonl \
    -output_dir DATASET_PATH
"""
4. Train the model with `train.py`

