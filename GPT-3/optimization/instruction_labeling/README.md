# Instruction Labeling

Instruction labeling means collecting the training label related to each instruction, \textit{i.e.}, check the actual downstream performance of each instruction on each training example.

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
