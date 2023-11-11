"""
Pre-process the dataset and save the processed version to disk
"""
import os
import random
import argparse
from transformers import T5TokenizerFast
from datasets import load_dataset

random.seed(46556)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-builder_script', type=str, default='Dataset.py',
                        help='Script that defines InstructionSelectionDataBuilder')
    parser.add_argument('-train_file', type=str, default='data/class_niv2_fewshot/train.jsonl',
                        help='Path to the training examples')
    parser.add_argument('-valid_file', type=str, default=None,
                        help='Path to the validation examples')
    parser.add_argument('-test_file', type=str, default=None,
                        help='Path to the test examples')
    parser.add_argument('-tokenizer_name', type=str, default='google/flan-t5-large')
    parser.add_argument('-output_dir', type=str, default='data/class_niv2_fewshot')
    parser.add_argument('-pad_to_multiple_of_8', default=False, action='store_true',
                        help='If true, pad the input_ids to a multiple of 8')
    args = parser.parse_args()

    tokenizer = T5TokenizerFast.from_pretrained(args.tokenizer_name)
    data_files = {"train": args.train_file}
    if args.valid_file is not None:
        data_files["validation"] = args.valid_file
    if args.test_file is not None:
        data_files["test"] = args.test_file

    print(f'Input max sequence length: {tokenizer.model_max_length}')

    datasets = load_dataset(
        args.builder_script,
        name="default",
        data_files=data_files,
        tokenizer=tokenizer,
        max_seq_length=tokenizer.model_max_length,
        pad_to_multiple_of_8=args.pad_to_multiple_of_8,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    datasets.save_to_disk(args.output_dir)


if __name__ == "__main__":
    main()
