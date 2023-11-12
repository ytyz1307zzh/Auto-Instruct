"""
Calculate evaluation scores (ROUGE/EM) for NIV2 tasks
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

import math
import json
import argparse
from Constants import (
    NIV2_TASK2METRIC,
    NIV2_OUT_DOMAIN_TEST_TASKS_SET2,
)
TEST_TASKS = NIV2_OUT_DOMAIN_TEST_TASKS_SET2


def find_example_task_name(data_id: str):
    for task in TEST_TASKS:
        if data_id.startswith(task):
            return task
    raise ValueError(f"Cannot find task name for data_id {data_id}")


def mean(array):
    assert isinstance(array, list)
    return sum(array) / len(array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-prediction', type=str, default='CKPT_NAME/predictions.json',
                        help='The file containing predictions of the trained model on dev or test set')
    parser.add_argument('-rankable_only', action='store_true', default=False,
                        help='If true, only calculate scores for rankable examples')
    parser.add_argument('-evaluation', type=str, choices=['rouge', 'task_specific'], default='rouge',
                        help='The evaluation metric to use. '
                             'rouge: all tasks use ROUGE, including classification. '
                             'task_specific: use the metric defined in NIV2_TASK2METRIC.')
    # parser.add_argument('-empty_baseline', type=int, default=0,
    #                     help='The index of the empty instruction baseline')
    # parser.add_argument('-naive_baseline', type=int, default=1,
    #                     help='The index of the naive instruction baseline')
    args = parser.parse_args()

    data = json.load(open(args.prediction, 'r', encoding='utf8'))
    print(f'Loaded {len(data)} predictions from {args.prediction}')
    results = {}
    task2metric = {}

    for data_id, example in data.items():
        task_name = find_example_task_name(data_id)
        select_instruction_idx = example['select_instruction']
        instruction_rouge_scores = example['instruction_rouge']

        if args.evaluation == 'task_specific':
            eval_metric = NIV2_TASK2METRIC[task_name]
        elif args.evaluation == 'rouge':
            eval_metric = 'ROUGE'
        else:
            raise ValueError(f"Unknown evaluation metric {args.evaluation}")
        task2metric[task_name] = eval_metric

        if task_name not in results.keys():
            results[task_name] = {"random": [], "select": [], "oracle": [], "naive": [], "worst": []}

        if eval_metric == "EM":
            instruction_scores = [1 if math.isclose(rouge_score, 1.0) else 0
                                  for rouge_score in instruction_rouge_scores]
        elif eval_metric == "ROUGE":
            instruction_scores = instruction_rouge_scores
        else:
            raise ValueError(f"Unknown evaluation metric {eval_metric}")

        if args.rankable_only:
            if math.isclose(max(instruction_scores), min(instruction_scores)):
                continue

        results[task_name]['random'].append(mean(instruction_scores))
        # results[task_name]['empty'].append(instruction_scores[args.empty_baseline])
        results[task_name]['naive'].append(instruction_scores[0])  # change the index if necessary
        results[task_name]['select'].append(instruction_scores[select_instruction_idx])
        results[task_name]['oracle'].append(max(instruction_scores))
        results[task_name]['worst'].append(min(instruction_scores))

    macro_average_random, macro_average_selected, macro_average_oracle = [], [], []
    macro_average_naive = []
    macro_average_worst = []
    task_cnt = 0
    for task_name, task_results in results.items():

        if len(task_results["select"]) == 0:
            continue

        print(f'Task: {task_name}, Metric: {task2metric[task_name]}, Examples: {len(task_results["select"])}, '
              f'Random instruction: {mean(task_results["random"]):.2%},'
              # f'Empty instruction: {mean(task_results["empty"]):.2%}, '
              f'Naive instruction: {mean(task_results["naive"]):.2%}, '
              f'Selected instruction: {mean(task_results["select"]):.2%}, '
              f'Oracle score: {mean(task_results["oracle"]):.2%}, '
              f'Worst score: {mean(task_results["worst"]):.2%}')
        task_cnt += 1

        macro_average_random.append(mean(task_results["random"]))
        macro_average_naive.append(mean(task_results["naive"]))
        macro_average_selected.append(mean(task_results["select"]))
        macro_average_oracle.append(mean(task_results["oracle"]))
        macro_average_worst.append(mean(task_results["worst"]))

    print(f'Total number of tasks: {task_cnt}')
    print(f'Macro average random instruction: {mean(macro_average_random):.2%}')
    print(f'Macro average naive instruction: {mean(macro_average_naive):.2%}')
    print(f'Macro average selected instruction: {mean(macro_average_selected):.2%}')
    print(f'Macro average oracle instruction: {mean(macro_average_oracle):.2%}')
    print(f'Macro average worst instruction: {mean(macro_average_worst):.2%}')


if __name__ == '__main__':
    main()

