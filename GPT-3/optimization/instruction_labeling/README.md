# Instruction Labeling

Instruction labeling means collecting the training label related to each instruction, \textit{i.e.}, check the actual downstream performance of each instruction on each training example.

To do this on a list of NIV2 tasks, call `niv2_label_instructions.py` with your OpenAI key, and it should automatically iterate through all tasks in the task directory you specify. This script actually iteratively calls `label_instructions.py` for each task. Check their command line arguments for more options.

Note: 
- The `instruction_file` argument refers to a JSON file where all generated instructions as well as the original seed instruction are collected into a single list (the file contains a list of 22 instructions).
- The saved label file (in JSONL) should contain the following contents:
    - The first element is the prompt template for the instruction ranking model
    - The second element is a list of candidate instructions
    - The third element is the list of few-shot demos
    - From the fourth element, there are all the training data as well as the labels to the instructions. In practice, we use `instruction_rouge` field for training the ranking model.