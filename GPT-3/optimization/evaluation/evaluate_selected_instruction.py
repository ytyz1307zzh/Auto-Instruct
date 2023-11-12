"""
Evaluate the selected instruction on ChatGPT
"""
import json
import os
import pdb
import sys
import math
import random
import shutil
import openai
import argparse
from tqdm import tqdm
from rouge_score import rouge_scorer
sys.path.append('GPT-3')
from utils.gpt_utils import chatgpt_single_turn_inference
from optimization.Constants import NIV2_TEST_TASK_FORMAT_INSTRUCTION
RougeScorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)


def read_jsonl_as_list(path: str):
    assert path.endswith('.jsonl')
    with open(path, 'r', encoding='utf8') as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    # print(f'Read {len(result)} data from {path}')
    return result


def save_list_as_jsonl(path: str, data):
    assert path.endswith('.jsonl')
    with open(path, 'w', encoding='utf8') as fout:
        for instance in data:
            fout.write(json.dumps(instance))
            fout.write('\n')
    print(f'Saved {len(data)} data to {path}')


def mean(array):
    assert isinstance(array, list)
    return sum(array) / len(array)


few_shot_prompt = """{{instruction}}

{{demo}}

Input: {{input}}
Output:"""

zero_shot_prompt = """{{instruction}}

Input: {{input}}
Output:"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-prediction', type=str, default='CKPT_NAME/test_predictions.json',
                        help='Path to the model prediction file')
    parser.add_argument('-test_data', type=str,
                        default='DATA_DIR/test.jsonl')
    parser.add_argument('-save_interval', type=int, default=20,
                        help='The interval of writing to output_file')
    parser.add_argument('-output_file', type=str,
                        default='CKPT_NAME/chatgpt_test_predictions_inference_results.jsonl')
    parser.add_argument('-model', type=str, default='gpt-3.5-turbo',
                        help='OpenAI model name')
    parser.add_argument('-max_tokens', type=int, default=80,
                        help='Max output sequence length')
    parser.add_argument('-timeout', type=int, default=20)
    parser.add_argument('-sleep', type=int, default=1)
    parser.add_argument('-api_key', type=str, required=True)
    parser.add_argument('-zero_shot', action='store_true', default=False,
                        help='If True, then conduct zero-shot inference')
    parser.add_argument('-random_baseline', action='store_true', default=False,
                        help='If True, randomly select an instruction from all candidate instructions '
                             'as the random instruction baseline')
    parser.add_argument('-naive_baseline', action='store_true', default=False,
                        help='If True, calculate the average score of the naive instruction')
    parser.add_argument('-append_format', action='store_true', default=False,
                        help='If True, append the formatting instruction to the end of the original instruction. '
                             'Usually used for zero-shot inference')
    args = parser.parse_args()
    assert not (args.random_baseline and args.naive_baseline)

    all_predictions = json.load(open(args.prediction, 'r', encoding='utf8'))
    test_data = read_jsonl_as_list(args.test_data)
    openai.api_key = args.api_key

    if os.path.exists(args.output_file):
        results = read_jsonl_as_list(args.output_file)
        remaining_data = test_data[len(results):]
        print(f'Read {len(results)} previous data, {len(remaining_data)} remaining...')
    else:
        results = []
        remaining_data = test_data

    add_format_tasks = set()

    for example in tqdm(remaining_data):
        id_ = example['id']
        input_text = example['input']
        target_text = example['target']
        task_name = '_'.join(id_.split('_')[:-1])

        instruction_list = example['instructions']
        demos = example['demos']

        prediction = all_predictions[id_]
        pred_instruction_idx = prediction['select_instruction']
        pred_scores = prediction['predict_scores']

        assert len(pred_scores) == len(instruction_list)
        selected_instruction = instruction_list[pred_instruction_idx]

        # For the random baseline
        if args.random_baseline:
            selected_instruction = random.choice(instruction_list)
        elif args.naive_baseline:
            selected_instruction = instruction_list[0] # change the index if necessary

        # For zero-shot prompting, add the formatting instructions to avoid free-form generation
        if args.append_format and task_name in NIV2_TEST_TASK_FORMAT_INSTRUCTION.keys():
            # For random selection, if we happens to select the naive instruction,
            # we don't need to add additional formatting instructions since it originally has
            if selected_instruction != instruction_list[0]:
                selected_instruction += '\n' + NIV2_TEST_TASK_FORMAT_INSTRUCTION[task_name]
            if task_name not in add_format_tasks:
                print(f'Add formatting instruction to {task_name}!')
            add_format_tasks.add(task_name)

        prompt_template = zero_shot_prompt if args.zero_shot else few_shot_prompt
        prompt = prompt_template.replace('{{instruction}}', selected_instruction)
        prompt = prompt.replace('{{demo}}', demos)
        prompt = prompt.replace('{{input}}', input_text)

        # This script is originally intended for ChatGPT, so change this call if using text-davinci-003
        output = chatgpt_single_turn_inference(
                inputs_with_prompts=prompt,
                model=args.model,
                max_tokens=args.max_tokens,
                num_return=1,
                temperature=0.0,
                top_p=0.0,
                stop=None,
                timeout=args.timeout,
                sleep=args.sleep,
            )

        output = output[0]
        score = RougeScorer.score(target=target_text.strip(), prediction=output.strip())['rougeL'].fmeasure

        results.append({
            "id": id_,
            "input": input_text,
            "target": target_text,
            "select_instruction": selected_instruction,
            "prediction": output,
            "score": score,
        })

        if len(results) % args.save_interval == 0:
            if os.path.exists(args.output_file):
                backup_file = args.output_file + '.bak'
                shutil.copy(args.output_file, backup_file)
                save_list_as_jsonl(args.output_file, results)
                os.remove(backup_file)
            else:
                save_list_as_jsonl(args.output_file, results)

    # collect scores for each task
    task2scores = {}
    for example in results:
        id_ = example['id']
        task_name = '_'.join(id_.split('_')[:-1])
        score = example['score']

        if task_name not in task2scores:
            task2scores[task_name] = []
        task2scores[task_name].append(score)

    # caclulate macro average score
    all_task_scores = []
    for task_name, score_list in task2scores.items():
        assert len(score_list) == 200 # Remove this if necessary
        all_task_scores.append(mean(score_list))
        print(f'{task_name}: ROUGE {mean(score_list):.2%}')

    print(f'Overall: ROUGE {mean(all_task_scores):.2%}')
    save_list_as_jsonl(args.output_file, results)
    print(f'Tasks where formatting instructions are added: {add_format_tasks} ({len(add_format_tasks)})')


if __name__ == '__main__':
    main()
