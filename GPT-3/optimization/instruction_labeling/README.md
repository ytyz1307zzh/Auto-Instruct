# Instruction Labeling

Instruction labeling means collecting the training label related to each instruction, *i.e.*, check the actual downstream performance of each instruction on each training example.

#### Label the instructions according to downstream performance

To do this on a list of NIV2 tasks, call `niv2_label_instructions.py` with your OpenAI key, and it should automatically iterate through all tasks in the task directory you specify. This script actually iteratively calls `label_instructions.py` for each task. Check their command line arguments for more options.

Note: 
- The `instruction_file` argument refers to a JSON file where all generated instructions as well as the original seed instruction are collected into a single list (the file contains a list of 22 instructions).
- The saved label file (in JSONL) should contain the following contents:
    - The first element is the prompt template for the instruction ranking model
    - The second element is a list of candidate instructions
    - The third element is the list of few-shot demos
    - From the fourth element, there are all the training data as well as the labels to the instructions.

#### Sample instructions for training

As mentioned in the Appendix A, to expedite training, we sample 8 instructions out of the total 22 for each training example. This is done in `niv2_relabel_rouge_and_pick_instructions.py`. This sampling is based on the popularity of each instruction, so that we downsample the "extraordinary" instructions to make a more balanced training distribution of instructions. We also conduct sampling on the level of examples so that examples with popular instructions will be less likely to be included in the training set. This also normalizes the ROUGE scores for the list-wise loss.

In practice, we use the following arguments to run this script:
`-sample_instructions 8 -min_qualify_range 10 -max_examples 400 -rank_criteria norm_score_max -pick_example_criteria instruction_popularity`

A example of training data:

```json
{
    "id": "task066-325aee69bf63414293c266c0154703fc",
    "input": "Sentence 1: I saw an old friend today. \n Sentence 3: We talked for hours about how we've been \n Sentence 4:  It was so nice to see my friend \n Sentence 5:  I can't wait for the next time we are able to catch up \n Given Sentence 2: He was never first for anything.",
    "target": "No",
    "instruction_idx": [6, 7, 9, 12, 13, 14, 19, 21],
    "instruction_labels": [0.0362745006332395, 0.0362745006332395, 0.0362745006332395, 0.0362745006332395, 0.0362745006332395, 0.0362745006332395, 0.7460784955673235, 0.0362745006332395],
    "instruction_outputs": [" Yes", " Yes", " Yes", " Yes", " Yes", " Yes", " No", " Yes"],
    "instruction_rouge": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]}
```

#### Merge all tasks into the final train/dev/test data files

In `mix_datasets.py`, we merge all data from each task into the final train/dev/test datasets. Detailed instruction can be found in its docstring. Every element in the final dataset will be in this format:

```json
    {
        "id": "task066_timetravel_binary_consistency_classification_task066-325aee69bf63414293c266c0154703fc",
        "input": "Sentence 1: I saw an old friend today. \n Sentence 3: We talked for hours about how we've been \n Sentence 4:  It was so nice to see my friend \n Sentence 5:  I can't wait for the next time we are able to catch up \n Given Sentence 2: He was never first for anything.",
        "target": "No",
        "instruction_idx": [6, 7, 9, 12, 13, 14, 19, 21],
        "instruction_labels": [0.0362745006332395, 0.0362745006332395, 0.0362745006332395, 0.0362745006332395, 0.0362745006332395, 0.0362745006332395, 0.7460784955673235, 0.0362745006332395],
        "instruction_rouge": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "prompt": "Example: {{input}}\n\nInstruction: {{instruction}}\n\nIs this a good instruction to solve the example?",
        "demos": [A STRING OF THE CONCATENATION OF FEW-SHOT DEMOS],
        "instructions": [A LIST OF ALL CANDIDATE INSTRUCTIONS, INCLUDING THE SEED INSTRUCTION]
    }
```
