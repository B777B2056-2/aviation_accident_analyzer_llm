#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from datasets import load_dataset
from transformers import AutoTokenizer


class NTSBAviationReportsInstructionDatasetBuilder(object):
  """"空难事故指令数据集构建器"""
  def __init__(self, json_file_path:str, tokenizer:AutoTokenizer):
    self.raw_dataset = load_dataset("json", data_files=json_file_path)
    self._build_tokenized_dataset(tokenizer)

  def _build_tokenized_dataset(self, tokenizer:AutoTokenizer) -> None:
    """构建huggingface数据集"""
    def format_prompt(data):
      """prompt格式化"""
      MAX_LENGTH = 65536
      instruction = tokenizer(
        f"<|im_start|>system\n{data['instruction']}<|im_end|>\n<|im_start|>user\n{data['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False
      )
      response = tokenizer(f"{data['output']}", add_special_tokens=False)
      input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
      attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # eos token补充为1
      labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

      if len(input_ids) > MAX_LENGTH:  # 截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

      return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
      }
    # 对每条数据应用格式化
    self.tokenized_dataset = self.raw_dataset.map(format_prompt, batched=False)

  def train_test_split(self, test_size:float):
    """划分数据集"""
    split_dataset = self.tokenized_dataset.train_test_split(test_size=test_size)
    train_dataset, test_dataset = split_dataset['train'], split_dataset['test']
    return train_dataset, test_dataset
