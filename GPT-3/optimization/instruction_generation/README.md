# Instruction Generation

1. `formulate_meta_prompt.py`: creates the meta-prompt for each task based on the meta-prompt templates in `GPT-3/optimization/instruction_generation_templates`, using the seed instruction and demonstration examples for each task.
2. `GPT-3/optimization/instruction_generation/gpt3_instruction_generation.py`: This scripts generates the candidate instructions for each task, with meta-prompts as input.
3. If using zero-shot setting, refer to `GPT-3/optimization/instruction_generation/zs_gpt3_instruction_generation.py`. The main difference is that this script adds the formatting instruction to the seed instruction for classification tasks, since we don't have few-shot demonstrations to help with the formatting in zero-shot.
