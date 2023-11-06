"""
Formulate the meta-prompt for instruction generation on FLAN collection datasets
"""

import os
import json
import argparse
from tqdm import tqdm
from typing import List, Dict


TEMPLATE_FILE_LIST = [
    'niv2_demo_task_with_demo_1.txt',
    'niv2_demo_task_with_demo_3.txt',
    'niv2_demo_task_with_demo_5.txt',
    'one-paragraph_instruction_with_demo.txt',
    'one-sentence_instruction_with_demo.txt',
    'step-by-step_instruction_with_demo.txt',
    'use-example_instruction_with_demo.txt'
]

ZS_TEMPLATE_FILE_LIST = [
    "zs_niv2_demo_task_1.txt",
    "zs_niv2_demo_task_3.txt",
    "zs_niv2_demo_task_5.txt",
    "zs_one-paragraph_instruction.txt",
    "zs_one-sentence_instruction.txt",
    "zs_step-by-step_instruction.txt"
]


def read_jsonl_as_list(path: str):
    assert path.endswith('.jsonl')
    with open(path, 'r', encoding='utf8') as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    # print(f'Read {len(result)} data from {path}')
    return result


def reform_data(example: Dict[str, str]):
    """
    Convert the examples from a dict format to a string format
    """
    input_text = example['input'].strip()
    output_text = example['target'].strip()
    example_string = f"Input: {input_text}\nOutput: {output_text}"
    return example_string


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task_dir', type=str, default='data/niv2_english',
                        help='Directory to NIV2 tasks')
    parser.add_argument('-naive_instruction_file', type=str, default='naive_instruction.txt')
    parser.add_argument('-demo_file', type=str, default='demo.jsonl')
    parser.add_argument('-template_dir', type=str, default='GPT-3/optimization/instruction_generation_templates',
                        help='Directory that contains templates for meta-prompts of instruction generation')
    parser.add_argument('-output_dir', type=str, default='data/niv2_english')
    parser.add_argument('-zero_shot', action='store_true', help='zero-shot setting indicator')
    args = parser.parse_args()
    all_tasks_dir = args.task_dir

    all_tasks_list = os.listdir(all_tasks_dir)
    delimiter = '\n\n'

    for task_name in tqdm(all_tasks_list):
        task_dir = os.path.join(all_tasks_dir, task_name)

        if not os.path.isdir(task_dir):
            continue

        # Get the seed instruction
        naive_instruction_file = os.path.join(task_dir, args.naive_instruction_file)
        naive_instruction = open(naive_instruction_file, 'r', encoding='utf8').read().strip()

        # Get the few-shot demo
        demo_file = os.path.join(task_dir, args.demo_file)
        demo_list = read_jsonl_as_list(demo_file)
        demo_list = [reform_data(example) for example in demo_list]

        template_file_list = TEMPLATE_FILE_LIST if not args.zero_shot else ZS_TEMPLATE_FILE_LIST

        for template_file_name in template_file_list:
            assert template_file_name.endswith('.txt')
            template_type_name = template_file_name[:-4]  # one-sentence_instruction

            with open(os.path.join(args.template_dir, template_file_name), 'r', encoding='utf8') as fin:
                template = fin.read().strip()

            template = template.replace("{{task}}", naive_instruction)
            template = template.replace("{{demos}}", delimiter.join(demo_list))

            os.makedirs(os.path.join(args.output_dir, task_name, template_type_name), exist_ok=True)
            output_path = os.path.join(args.output_dir, task_name, template_type_name, 'meta_prompt.txt')
            with open(output_path, 'w', encoding='utf8') as fout:
                fout.write(template)


if __name__ == "__main__":
    main()
