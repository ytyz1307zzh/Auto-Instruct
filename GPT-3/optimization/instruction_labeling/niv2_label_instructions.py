"""
Iteratively call "label_instructions.py" on NIV2 tasks
"""

import os
import sys
import platform
if platform.platform().startswith("Windows"):
    path = os.path.abspath(__file__)
    sys.path.append('\\'.join(path.split('\\')[:-2]))
else:
    path = os.path.abspath(__file__)
    sys.path.append('/'.join(path.split('/')[:-2]))

import json
import argparse

EVAL_METRICS = [
    "exact_match",
    "norm_exact_match",
    "prefix_match",
    "norm_prefix_match",
    "rouge_L",
    "relative_rouge_L"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task_dir', type=str, default='data/niv2_english',
                        help='Path to the directory containing all tasks')
    # Following file arguments are the same with the ones in "label_instructions.py"
    parser.add_argument('-instruction_file', type=str, default='text003_all_prompts_22.json')
    parser.add_argument('-data_file', type=str, default='all_examples.jsonl')
    parser.add_argument('-demo_file', type=str, default='demo.jsonl')
    parser.add_argument('-gpt_inference_prompt_file', type=str, default='fs_downstream_inference_prompt_template.txt')
    parser.add_argument('-demo_template_file', type=str, default='demo_template.json')
    parser.add_argument('-instruction_selection_prompt_file', type=str,
                        default='instruction_selection_prompt.txt')
    parser.add_argument('-temp_save_file', type=str, default='temp_save.jsonl',
                        help='File to save intermediate results')
    parser.add_argument('-instruction_group_file', type=str, default=None,
                        help='If not None, then sample instructions based on the groups. '
                             'Instructions will not be sampled from the same group. '
                             'args.sample_instructions must not be None.')
    parser.add_argument('-model', type=str, default='text-davinci-003',
                        help='Which GPT-3 model to use')
    parser.add_argument('-batch_size', type=int, default=20,
                        help='How many prompts to send together to GPT3')
    parser.add_argument('-sample_instructions', type=int, default=None,
                        help='If not None, randomly sample a subset of instructions for each example')
    parser.add_argument('-sample_data', type=int, default=None,
                        help='If not None, only use the first sample_data examples')
    parser.add_argument('-max_tokens', type=int, default=128,
                        help='The max number of tokens for GPT generation')
    parser.add_argument('-timeout', type=int, default=20,
                        help='If API respond time exceeds this value, then call it again')
    parser.add_argument('-sleep', type=int, default=1,
                        help='How long to sleep if rate limit is reached')
    parser.add_argument('-api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('-output_file', type=str, default='test_instruction_labels_text003.jsonl',
                        help='The file to save instruction labeling results')
    parser.add_argument('-eval_metric', type=str, choices=EVAL_METRICS, default='rouge_L')

    # async related arguments
    parser.add_argument('-use_async', action='store_true', default=False, help='Whether to use async API call')
    parser.add_argument('-num_async_workers', default=None, type=int,
                        help='If use async API, how many async jobs to create in parallel.')
    args = parser.parse_args()

    task_names = os.listdir(args.task_dir)

    for i, task in enumerate(task_names):

        if not os.path.isdir(os.path.join(args.task_dir, task)):
            continue

        output_file_path = os.path.join(args.task_dir, task, args.output_file)

        if not os.path.exists(os.path.join(args.task_dir, task, args.instruction_file)):
            continue
        
        options = f"-task {task} " \
                  f"-instruction_file {os.path.join(args.task_dir, task, args.instruction_file)} " \
                  f"-data_file {os.path.join(args.task_dir, task, args.data_file)} " \
                  f"-demo_file {os.path.join(args.task_dir, task, args.demo_file)} " \
                  f"-gpt_inference_prompt_file {os.path.join(args.task_dir, task, args.gpt_inference_prompt_file)} " \
                  f"-demo_template_file {os.path.join(args.task_dir, task, args.demo_template_file)} " \
                  f"-instruction_selection_prompt_file {os.path.join(args.task_dir, task, args.instruction_selection_prompt_file)} " \
                  f"-temp_save_file {os.path.join(args.task_dir, task, args.temp_save_file)} " \
                  f"-output_file {output_file_path} " \
                  f"-model {args.model} " \
                  f"-batch_size {args.batch_size} " \
                  f"-max_tokens {args.max_tokens} " \
                  f"-timeout {args.timeout} " \
                  f"-sleep {args.sleep} " \
                  f"-api_key {args.api_key} " \
                  f"-eval_metric {args.eval_metric} "

        if args.sample_instructions is not None:
            options += f"-sample_instructions {args.sample_instructions} "
            if args.instruction_group_file is not None:
                options += f"-instruction_group_file {os.path.join(args.task_dir, task, args.instruction_group_file)} "
        if args.sample_data is not None:
            options += f"-sample_data {args.sample_data} "

        if args.use_async:
            options += f"-use_async "
            assert args.num_async_workers is not None
            options += f"-num_async_workers {args.num_async_workers} "

        task_command = f"python GPT-3/optimization/instruction_labeling/label_instructions.py {options}"
        print('\n' + task_command, end='\n\n')
        print("*" * 10 + task + f" ({i + 1}/{len(task_names)})" + "*" * 10)
        # If the output file exists, skip this task
        if os.path.exists(output_file_path):
            print(f"Output file {output_file_path} exists, skip this task...")
            continue
        os.system(task_command)

        # Remove the temporary save file if it is empty (caused by an interrupted run)
        temp_save_file = os.path.join(args.task_dir, task, args.temp_save_file)
        if os.path.exists(temp_save_file) and os.stat(temp_save_file).st_size == 0:
            os.remove(temp_save_file)

        # After the program exits, if the output file does not exist, meaning the program does not finish normally
        if not os.path.exists(output_file_path):
            break


if __name__ == '__main__':
    main()
