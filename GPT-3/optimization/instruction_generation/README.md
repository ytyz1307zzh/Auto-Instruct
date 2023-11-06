# Instruction Generation

1. `formulate_meta_prompt.py`: creates the meta-prompt for each task based on the meta-prompt templates in `GPT-3/optimization/instruction_generation_templates`, using the seed instruction and demonstration examples for each task.
2. `GPT-3/optimization/instruction/generation/gpt3_instruction_generation.py`: This scripts generates the candidate instructions for each task, with meta-prompts as input.
