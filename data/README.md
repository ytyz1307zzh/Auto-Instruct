# NIV2 (Natural Instructions V2) data

This directory contains the NIV2 data used in training / testing our model. For the train/test split, please refer to `GPT-3/optimization/Constants.py`

In each task directory, there are the following files:

- `all_examples.jsonl`: Examples extracted from the original NIV2 collection, up to 5k for each task. These are where the training examples come from.
- `train_examples.jsonl` (for training tasks): The examples that are used to train our model. This is maximum 400 examples for each task. The number of examples is affected by the cost of labeling instructions, the concern of balancing data size across tasks, and also the exclusion of instruction-insensitive examples (examples where different instructions lead to the same result).
- `test_examples_200.jsonl` (for test tasks): Examples (200 for each task) for testing.
- `task_metadata.json`: the metadata of each task.
- `naive_instruction.txt`: the basic human-written seed instruction used in instruction generation.
- `demo.jsonl`: the 3-shot demonstration examples in few-shot in-context learning.

The following files should be the same for each task:
- `demo_template.json`: the template of formatting each demonstration when contructing the prompt.
- `instruction_selection_prompt.txt`: the input template when training FLAN-T5.
- `fs_downstream_inference_prompt_template.txt`: the template of prompting GPT-3 in few-shot setting.
- `zs_downstream_inference_prompt_template.txt`: the template of prompting GPT-3 in zero-shot setting.
