"""
Data collator and data loader
"""

import torch
from typing import List, Dict, Any
from dataclasses import dataclass
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import PretrainedConfig


@dataclass
class InstructionSelectionCollator:
    """
    Data Collator for instruction selection
    """
    config: PretrainedConfig

    def __post_init__(self):
        pass

    def __call__(self, example_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        :param examples: A list of items in the dataset. Each item is a dictionary of input fields
        """
        string_attributes = ["id"]
        tensor_attributes_to_pad_3d = ["input_ids", "decoder_input_ids"]
        tensor_attributes_2d = ["instruction_labels", "instruction_rouge"]
        batch = {}

        for key in example_list[0].keys():
            value_list = [example[key] for example in example_list]  # (batch, num_instructions, seq_length)

            if key in string_attributes:
                batch[key] = value_list

            elif key in tensor_attributes_to_pad_3d:
                value_list = self.dynamic_pad_to_max_length(value_list)
                batch_tensor = torch.tensor(value_list)
                batch[key] = batch_tensor

            elif key in tensor_attributes_2d:
                batch_tensor = torch.tensor(value_list)
                batch[key] = batch_tensor

            else:
                raise ValueError(f"Invalid data attribute {key}!")

        # input_ids: (batch, num_instructions, input_seq_length)
        # output_ids: (batch, num_instructions, output_seq_length)

        return batch

    def dynamic_pad_to_max_length(self, value_list):
        """
        value_list: input_ids or output_ids, shape (batch, num_instructions, seq_length)
        """
        # Find the max sequence length
        max_seq_length = 0
        for example in value_list:
            for instruction in example:
                max_seq_length = max(max_seq_length, len(instruction))

        # If all sequences are padded to the multiple of 8 during dataset construction,
        # then we don't need to do anything here because the max_seq_length must be a multiple of 8.
        for i in range(len(value_list)):
            for j in range(len(value_list[i])):
                value_list[i][j] += [self.config.pad_token_id] * (max_seq_length - len(value_list[i][j]))

        return value_list


class InstructionSelectionDataLoader(DataLoader):

    def __init__(self, args, dataset, collate_fn, is_training):
        if is_training:
            sampler = RandomSampler(dataset) if args.world_size == 1 else DistributedSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler = SequentialSampler(dataset) if args.world_size == 1 else \
                DistributedSampler(dataset, shuffle=False)
            batch_size = args.test_batch_size
        super(InstructionSelectionDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn)
