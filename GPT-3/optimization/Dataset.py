"""
The Pytorch Dataset class for instruction selection
"""

import os
import json
import pdb
import torch
import random
import logging
import datasets
import argparse
from copy import deepcopy
from typing import Dict, List

from Logger import logger
from data_utils import read_jsonl_as_list
from datasets.download.download_manager import DownloadManager


class InstructionSelectionDataConfig(datasets.BuilderConfig):

    def __init__(
        self,
        tokenizer,
        max_seq_length: int,
        pad_to_multiple_of_8: bool,
        decoder_start_token_id: int,
        **kwargs
    ):
        super(InstructionSelectionDataConfig, self).__init__(**kwargs)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_to_multiple_of_8 = pad_to_multiple_of_8
        self.decoder_start_token_id = decoder_start_token_id


class InstructionSelectionDataBuilder(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        InstructionSelectionDataConfig(
            name="default",
            tokenizer=None,
            max_seq_length=0,
            pad_to_multiple_of_8=False,
            decoder_start_token_id=0
        )
    ]

    # Two splits in total: train and eval
    split_names = [datasets.Split.TRAIN]

    def _info(self):
        return datasets.DatasetInfo(
            description="Instruction Selection Dataset: select the best instruction for each example",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "input_ids": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("int64"))),
                    "decoder_input_ids": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("int64"))),
                    'instruction_labels': datasets.features.Sequence(datasets.Value("float")),
                    "instruction_rouge": datasets.features.Sequence(datasets.Value("float")),
                }
            )
        )

    def _split_generators(self, dl_manager: DownloadManager):
        if self.config.data_dir:
            assert os.path.isdir(self.config.data_dir)
        if self.config.data_files:
            assert len(self.config.data_files["train"]) == 1
            assert os.path.isfile(self.config.data_files["train"][0])

            if "validation" in self.config.data_files.keys():
                assert len(self.config.data_files["validation"]) == 1
                assert os.path.isfile(self.config.data_files["validation"][0])
                self.split_names.append(datasets.Split.VALIDATION)

            if "test" in self.config.data_files.keys():
                assert len(self.config.data_files["test"]) == 1
                assert os.path.isfile(self.config.data_files["test"][0])
                self.split_names.append(datasets.Split.TEST)

        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "data_dir": self.config.data_dir,
                    "data_files": self.config.data_files,
                    "split_name": split,
                    "tokenizer": self.config.tokenizer,
                    "max_seq_length": self.config.max_seq_length,
                    "pad_to_multiple_of_8": self.config.pad_to_multiple_of_8,
                    "decoder_start_token_id": self.config.decoder_start_token_id
                }
            )
            for split in self.split_names
        ]

    def _generate_examples(
        self,
        data_dir: str,
        data_files: Dict[str, str],
        split_name: str,
        tokenizer,
        max_seq_length: int,
        pad_to_multiple_of_8: bool,
        decoder_start_token_id: int
    ):
        if split_name == "train":
            data_path = str(data_files["train"][0])
        elif split_name == "validation":
            data_path = str(data_files["validation"][0])
        elif split_name == "test":
            data_path = str(data_files["test"][0])
        else:
            raise ValueError(f"Unsupported split_name {split_name}.")

        dataset = read_jsonl_as_list(data_path)
        # if split_name == "train":
        #     random.shuffle(dataset)

        for idx in range(len(dataset)):
            example = dataset[idx]
            id_ = example["id"]
            input_text = example["input"]
            prompt_template = example["prompt"]
            demo = example["demos"]  # a string of all demos putting together
            # TODO: the following line has bugs. In some tasks, demo itself has '\n\n' in it.
            #  If the instruction selection template includes the demo, you need to fix the way to split the demo.
            # demo_list = demo.strip().split('\n\n')
            demo_list = []

            instruction_list = example["instructions"]
            instruction_labels = example['instruction_labels']
            instruction_rouge_scores = example['instruction_rouge']
            num_instructions = len(instruction_list)
            # instruction_labels = example["instruction_labels"]

            input_prompt = prompt_template.replace("{{input}}", input_text)

            all_input_ids = []
            for instruction in instruction_list:
                input_prompt_with_instruction = input_prompt.replace("{{instruction}}", instruction)
                input_prompt_with_instruction_demo = input_prompt_with_instruction.replace("{{demo}}", demo)

                length_qualified = False
                truncated_demo_list = deepcopy(demo_list)
                while not length_qualified:
                    tokenization_result = tokenizer(
                        input_prompt_with_instruction_demo,
                        add_special_tokens=True,
                        padding=False,
                        truncation=False
                    )
                    if len(tokenization_result.input_ids) > max_seq_length and len(truncated_demo_list) > 0:
                        truncated_demo_list = truncated_demo_list[:-1]
                        if len(truncated_demo_list) > 0:
                            truncated_demo = '\n\n'.join(truncated_demo_list)
                        else:
                            truncated_demo = 'N/A'
                        input_prompt_with_instruction_demo = \
                            input_prompt_with_instruction.replace("{{demo}}", truncated_demo)
                    else:
                        length_qualified = True

                input_ids = tokenization_result.input_ids
                all_input_ids.append(input_ids)

            # pad inputs of different instructions to the same length
            batch_seq_length = max([len(input_ids) for input_ids in all_input_ids])
            # let batch_seq_length to be the closest multiple of 8
            if pad_to_multiple_of_8:
                batch_seq_length = (batch_seq_length + 7) // 8 * 8

            all_input_ids = [
                tokenizer.prepare_for_model(
                    input_ids,
                    add_special_tokens=False,  # already added in previous step
                    padding='max_length',  # fixed-length padding
                    truncation=True,  # If reducing the number of demos cannot solve the issue of overflowing,
                                      # then truncate the input sequence
                    max_length=batch_seq_length,
                    return_tensors='pt'
                ).input_ids
                for input_ids in all_input_ids
            ]
            all_input_ids = torch.stack(all_input_ids, axis=0)

            # instruction_outputs = ["yes" if label == 1 else "no" for label in instruction_labels]
            # output_ids = tokenizer(
            #     instruction_outputs,
            #     add_special_tokens=True,
            #     padding=True,
            #     return_tensors='pt'
            # ).input_ids

            # Add decoder start tokens (<pad> in FLAN-T5) to the decoder input
            decoder_start_ids = torch.full((num_instructions,), decoder_start_token_id)
            # output_ids = torch.cat((torch.unsqueeze(decoder_start_ids, dim=1), output_ids), dim=1)

            # instruction_labels: the training labels for instruction selection
            # instruction_rouge: used for evaluation, the rouge score of each instruction output
            item = {
                "id": id_,
                "input_ids": all_input_ids,
                "decoder_input_ids": torch.unsqueeze(decoder_start_ids, dim=1),
                'instruction_labels': torch.Tensor(instruction_labels),  # (num_instructions,)
                'instruction_rouge': torch.Tensor(instruction_rouge_scores)  # (num_instructions,)
            }

            yield id_, item
