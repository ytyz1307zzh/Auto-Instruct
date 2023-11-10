"""
Mix datasets from given directories into a one big dataset
- For training set: mix all data from args.train_data_file of each task
- For dev set:
    - If you want to use a seen dev set (examples from training tasks, but not in the training set), use args.max_task_size, args.split_seen_dev, and args.seen_dev_size (in the paper we used seen dev)
    - If you want to use an unseen dev set (examples from tasks that are not included in either training or testing), specify args.dev_data_file and args.split_unseen_dev, and also you need to specify DEV_TASKS global variable by yourself
- For test set: specify args.test_set and pass the test data file name through args.train_data_file
"""

import os
import pdb
import sys
import platform
if platform.platform().startswith("Windows"):
    path = os.path.abspath(__file__)
    sys.path.append('\\'.join(path.split('\\')[:-2]))
else:
    path = os.path.abspath(__file__)
    sys.path.append('/'.join(path.split('/')[:-2]))

import json
import math
import random
import string
import argparse
from copy import deepcopy
from tqdm import tqdm
from typing import List, Union
from Constants import (
    NIV2_OUT_DOMAIN_TRAIN_TASKS_SET2,
)
TRAIN_TASKS = NIV2_OUT_DOMAIN_TRAIN_TASKS_SET2
# TODO: you have to change DEV_TASKS if you want to use unseen dev tasks
DEV_TASKS = NIV2_OUT_DOMAIN_TRAIN_TASKS_SET2

random.seed(42)


def read_jsonl_as_list(path: str):
    assert path.endswith('.jsonl')
    with open(path, 'r', encoding='utf8') as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    # print(f'Read {len(result)} data from {path}')
    return result


def save_list_as_jsonl(path: str, data: List):
    assert path.endswith('.jsonl')
    with open(path, 'w', encoding='utf8') as fout:
        for instance in data:
            fout.write(json.dumps(instance))
            fout.write('\n')
    # print(f'Saved {len(data)} data to {path}')


def mix_all_datasets(
    data_file_list: List[str],
    max_task_size: Union[int, None],
    min_task_size: Union[int, None],
    max_dev_size: Union[int, None],
    upsample: bool
):
    mixed_train_data = []
    mixed_dev_data = []

    for data_path in data_file_list:
        task_name = data_path.strip().split('/')[-2]

        data = read_jsonl_as_list(data_path)
        prompt = data[0]['prompt']
        instruction_list = data[1]['instructions']
        demos = data[2]['demos']

        data = data[3:]
        task_data = []
        for example in data:
            # example keys: "id", "input", "target", "instruction_labels", "instruction_idx", "instruction_outputs"
            example['prompt'] = prompt
            example["id"] = f"{task_name}_{example['id']}"
            example["demos"] = demos
            example["instructions"] = [instruction_list[idx] if len(instruction_list[idx]) > 0 else "N/A"
                                       for idx in example["instruction_idx"]]
            # example.pop("instruction_idx", None)
            example.pop("instruction_outputs", None)
            example.pop('relative_rouge', None)

            task_data.append(example)

        if min_task_size is not None and len(task_data) < min_task_size:
            continue

        if max_task_size is not None and len(task_data) > max_task_size:
            dev_data = deepcopy(task_data[max_task_size:(max_task_size + max_dev_size)])
            task_data = task_data[:max_task_size]
        else:
            dev_data = []

        if upsample and len(task_data) < max_task_size:
            num_replicate = max_task_size // len(task_data) + 1  # the number of replicates needed for upsampling
            # For each upsampled replicate, deepcopy it to prevent mixing up the IDs
            for _ in range(num_replicate):
                task_data.extend(deepcopy(task_data))
            task_data = task_data[:max_task_size]

            for example in task_data:
                # generate a random 5-character string identifier to avoid duplicate ids
                random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
                example['id'] = f"{example['id']}_{random_id}"

        mixed_train_data.extend(deepcopy(task_data))
        mixed_dev_data.extend(dev_data)
        print(f'{task_name} finished: {len(task_data)} examples')
        if max_dev_size > 0:
            print(f'Used as dev set: {len(dev_data)} examples')

    return mixed_train_data, mixed_dev_data


def main():
    parser = argparse.ArgumentParser()
    # data/niv2_english
    parser.add_argument('-data_dir', type=str, nargs="+", required=True,
                        help='Input directories, can be multiple')
    parser.add_argument('-train_data_file', type=str, default='train_instruction_labels_text003_select.jsonl',
                        help='Path to the data file of each training task')
    parser.add_argument('-dev_data_file', type=str, default='train_instruction_labels_text003_select.jsonl',
                        help='Path to the data file of each dev task (for unseen dev)')
    parser.add_argument('-output_dir', type=str, default='data/class_niv2_fewshot',
                        help='Output directory')
    parser.add_argument('-max_task_size', type=int, default=None,
                        help='Max number of examples per task')
    parser.add_argument('-min_task_size', type=int, default=None,
                        help='If specified, tasks with fewer examples than this number will be skipped')
    parser.add_argument('-train_upsample', action='store_true', default=False,
                        help='If true, upsample the training data to args.max_task_size')
    parser.add_argument('-split_seen_dev', action='store_true', default=False,
                        help='If true, separate some data from the training set as dev set. '
                             'Only effective if args.max_task_size is specified.')
    parser.add_argument('-seen_dev_size', type=int, default=0,
                        help='The number of examples in training tasks to use as seen dev set.')
    parser.add_argument('-split_unseen_dev', action='store_true', default=False,
                        help='If true, split all tasks into train and dev tasks; '
                             'if processing the test set, set this to False')
    parser.add_argument('-dev_rankable_only', action='store_true', default=False,
                        help='If true, only include rankable examples in the dev set')
    parser.add_argument('-test_set', action='store_true', default=False,
                        help='If true, process the test set instead of the training set')
    args = parser.parse_args()

    train_data_file_list, dev_data_file_list = [], []
    os.makedirs(args.output_dir, exist_ok=True)
    
    for source_data_dir in args.data_dir:
        task_list = os.listdir(source_data_dir)

        for task in task_list:
            if not os.path.isdir(os.path.join(source_data_dir, task)):
                continue

            # unseen dev
            if args.split_unseen_dev and task in DEV_TASKS:
                data_file_path = os.path.join(source_data_dir, task, args.dev_data_file)
                if not os.path.exists(data_file_path):
                    print(f'File {args.dev_data_file} does not exist in task {task}')
                    continue
                dev_data_file_list.append(data_file_path)
            # training / seen dev / test
            elif task in TRAIN_TASKS:
                data_file_path = os.path.join(source_data_dir, task, args.train_data_file)
                if not os.path.exists(data_file_path):
                    print(f'File {args.train_data_file} does not exist in task {task}')
                    continue
                train_data_file_list.append(data_file_path)
            else:
                continue

    if args.split_seen_dev:
        max_dev_size = args.seen_dev_size
    else:
        max_dev_size = 0

    if args.test_set:
        print(f'\nProcessing test data...')
        test_data, _ = mix_all_datasets(
            train_data_file_list,
            max_task_size=None,
            min_task_size=None,
            max_dev_size=0,
            upsample=False
        )
        print(f'Test data finished. {len(test_data)} examples in total.')
        save_list_as_jsonl(os.path.join(args.output_dir, 'test.jsonl'), test_data)
    else:
        print(f'\nProcessing training data...')
        train_data, seen_dev_data = mix_all_datasets(
            train_data_file_list,
            max_task_size=args.max_task_size,
            min_task_size=args.min_task_size,
            max_dev_size=max_dev_size,
            upsample=args.train_upsample
        )
        print(f'Training data finished. {len(train_data)} examples in total.')
        save_list_as_jsonl(os.path.join(args.output_dir, 'train.jsonl'), train_data)
        if len(seen_dev_data) > 0:
            print(f'Seen dev data finished. {len(train_data)} examples in total.')
            save_list_as_jsonl(os.path.join(args.output_dir, 'seen_dev.jsonl'), seen_dev_data)

        if len(dev_data_file_list) > 0:
            print(f'\nProcessing dev data...')
            dev_data, _ = mix_all_datasets(
                dev_data_file_list,
                max_task_size=None,
                min_task_size=None,
                max_dev_size=0,  # this is for seen dev, nothing to do with unseen dev
                upsample=False
            )

            # remove unrankable examples in dev data
            if args.dev_rankable_only:
                rankable_dev_data = []
                for example in dev_data:
                    instruction_rouge = example['instruction_rouge']
                    if math.isclose(max(instruction_rouge), min(instruction_rouge)):
                        continue
                    else:
                        rankable_dev_data.append(example)
            else:
                rankable_dev_data = dev_data

            print(f'Unseen dev data finished. {len(dev_data)} examples in total, {len(rankable_dev_data)} are rankable.')
            save_list_as_jsonl(os.path.join(args.output_dir, 'unseen_dev.jsonl'), rankable_dev_data)

    print(f'Training tasks: {len(train_data_file_list)}')


if __name__ == '__main__':
    main()
