#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import argparse
from transformers import AutoTokenizer
from transformers import pipeline, Pipeline
from peft import AutoPeftModelForCausalLM


def parse_inference_arguments():
  parser = argparse.ArgumentParser(description="模型推理参数")
  parser.add_argument('--tokenizer_path', type=str, default="models/tokenizer", help='分词器存放路径（可选）')
  parser.add_argument('--model_path', type=str, default="models/sft", help='sft产出模型存放路径')
  return parser.parse_args()


def build_pipeline(args) -> Pipeline:
  """构建推理用pipeline"""
  # 加载分词器
  tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="Qwen/Qwen2-0.5B",
    cache_dir=args.tokenizer_path
  )
  tokenizer.padding_side = "left"

  # 加载SFT模型
  model = AutoPeftModelForCausalLM.from_pretrained(
    args.model_path,
    device_map="auto",
  )

  # 构建pipeline
  return pipeline(task="text-generation", model=model, tokenizer=tokenizer)

def get_generation_parameters():
    """获取用户输入的生成参数"""
    params = {
      "max_new_tokens": int(
        input("Enter max new tokens [4096]: ") or 4096
      ),
      "temperature": float(
        input("Enter temperature [1.0]: ") or 1.0
      ),
      "top_p": float(
        input("Enter top_p [1.0]: ") or 1.0
      ),
      "repetition_penalty": float(
        input("Enter repetition penalty [1.0]: ") or 1.0
      ),
      "length_penalty": float(
        input("Enter length penalty [1.0]: ") or 1.0
      ),
      "num_beams": int(
        input("Enter number of beams [1]: ") or 1
      )
    }

    sampling_choice = input("Use sampling? (y/n) [y]: ").lower() or 'y'
    params["do_sample"] = sampling_choice == 'y'

    return params


def inference_once(generator:Pipeline) -> bool:
  """单次推理"""
  params = get_generation_parameters()
  context = input("Enter context: ")
  prompt = input("Enter prompt: ")
  if prompt == "exit":
    return False
  full_prompt = f"<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
  result = generator(full_prompt, **params)
  if len(result) == 0:
    print("No answer")
  else:
    result = result[0]["generated_text"]
    answer = result.split("<|im_start|>assistant\n")[1]
    print(f"Answer: {answer}")
  return True


if __name__ == "__main__":
  pipe = build_pipeline(args=parse_inference_arguments())
  while True:
    needNext = inference_once(pipe)
    if not needNext:
      break
