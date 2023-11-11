import os
import json
import time
from Logger import logger
from typing import List


def read_jsonl_as_list(path: str):
    assert path.endswith('.jsonl')
    with open(path, 'r', encoding='utf8') as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    print(f'Read {len(result)} data from {path}')
    return result


def save_list_as_jsonl(path: str, data: List):
    assert path.endswith('.jsonl')
    with open(path, 'w', encoding='utf8') as fout:
        for instance in data:
            fout.write(json.dumps(instance))
            fout.write('\n')
    print(f'Saved {len(data)} data to {path}')


def save_predictions_to_json(
    data_ids,
    predict_positive_scores,
    select_instruction_indices,
    instruction_rouge_scores,
    save_path
):
    prediction_dict = {}
    for id_, predict_scores, select_index, instruction_scores in zip(
            data_ids, predict_positive_scores, select_instruction_indices, instruction_rouge_scores):
        prediction_dict[id_] = {
            "predict_scores": predict_scores,
            "select_instruction": select_index,
            "instruction_rouge": instruction_scores
        }

    with open(save_path, "w", encoding='utf8') as f:
        json.dump(prediction_dict, f)


def combine_prediction_files(output_dir, world_size, prefix):
    all_results = {}
    file_list = [os.path.join(output_dir, "{}predictions_ps{}.json".format(prefix, gpu_number))
                 for gpu_number in range(world_size)]

    for filename in file_list:
        result = json.load(open(filename, 'r', encoding='utf8'))
        print(f'Loaded {len(result)} predictions from {filename}.')
        all_results.update(result)

    save_path = os.path.join(output_dir, "{}predictions.json".format(prefix))
    json.dump(all_results, open(save_path, 'w', encoding='utf8'), ensure_ascii=False)
    print(f'Saved {len(all_results)} predictions to {save_path}.')
    for filename in file_list:
        os.remove(filename)


def save_checkpoint(model, save_dir: str):
    for i in range(10):
        try:
            model.save_pretrained(save_dir)
            break
        except OSError:
            time.sleep(30)
            continue
