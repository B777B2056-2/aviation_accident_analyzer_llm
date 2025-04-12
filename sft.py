#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import argparse
from data.preprocess import NTSBAviationReportsInstructionDatasetBuilder
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq
import numpy as np
import evaluate


MODEL_NAME = "Qwen/Qwen2-0.5B"


def load_lora_model():
  """
    构建LoRA模型结构
    原参数矩阵为w，lora旁路输出为A * B，缩放因子为lora_alpha，lora输出则为w + (lora_alpha / r) × (A * B)
    设w.shape = [d, k]，则A.shape = [d, r], B.shape = [r, k], 其中r为所谓的秩大小
  """
  global MODEL_NAME
  # 获取预训练模型
  pretrained_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
  peft_conf = LoraConfig(
    r=64,
    lora_alpha=32,  # lora缩放因子，用于控制lora旁路对原参数矩阵的影响
    target_modules=["q_proj", "k_proj", "v_proj"],  # lora作用到的模型内模块
    lora_dropout=0.1,  # lora旁路的dropout
    bias="none",  # lora旁路是否需要bias
    task_type="CAUSAL_LM",
  )
  lora_model = get_peft_model(pretrained_model, peft_conf)
  return lora_model, peft_conf


def compute_metrics(eval_predictions):
  """评估指标计算"""
  metric = evaluate.load("glue", "mrpc")
  logits, labels = eval_predictions
  predictions = np.argmax(logits, axis=-1)
  return metric.compute(predictions=predictions, references=labels)


def sft(lora_model, peft_conf, sft_tokenizer, num_train_epochs:int, model_path:str):
  """SFT微调"""
  sft_tokenizer.padding_side = "left"  # 目的是使得推理时attention处理方式与训练时一致
  # SFT微调参数配置
  sft_config = SFTConfig(
    output_dir=model_path,
    seed=3407,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=8,
    # gradient_accumulation_steps=4,  # batch_size = per_device_train_batch_size * gradient_accumulation_steps
    learning_rate=2e-4,
    lr_scheduler_type="cosine",  # 学习率自动调整的策略
    optim="adamw_torch",
    bf16=True,  # bf16+fp32混合精度训练
    dataset_text_field='text',
  )
  # 开始SFT微调
  trainer = SFTTrainer(
    model=lora_model,
    args=sft_config,
    data_collator=DataCollatorForSeq2Seq(tokenizer=sft_tokenizer),
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_conf,
    compute_metrics=compute_metrics,
  )
  trainer.train()
  # 保存模型
  trainer.model.save_pretrained(os.path.join(model_path, "sql_generator"))


def parse_sft_arguments():
  parser = argparse.ArgumentParser(description="模型SFT参数")
  parser.add_argument('--n_epoch', type=int, default=5, help='训练轮次')
  parser.add_argument('--raw_json_path', type=str, default="data/instructions/all.json", help='原始json路径')
  parser.add_argument('--test_size', type=float, default=0.1, help='测试集占比')
  parser.add_argument('--model_path', type=str, default="models/sft", help='sft产出模型存放路径')
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_sft_arguments()
  # 下载分词器
  tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=MODEL_NAME,
    cache_dir=os.path.join("../models", "tokenizer")
  )
  print("tokenizer download done, start building dataset...")
  # 构建数据集
  dataset_builder = NTSBAviationReportsInstructionDatasetBuilder(
    json_file_path=args.raw_json_path,
    tokenizer=tokenizer
  )
  print("dataset built done, start splitting dataset...")
  train_dataset, test_dataset = dataset_builder.train_test_split(test_size=args.test_size)
  # 开始微调
  print("dataset split done, start training...")
  model, peft_config = load_lora_model()
  sft(model, peft_config, tokenizer, model_path=args.model_path, num_train_epochs=args.n_epoch)
  print("done")
