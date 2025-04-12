#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import argparse
import os
import fitz
import threading
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, wait


QUESTION_ANSWER_GENERATE_PROMPT_TEMPLATE = '''
# Role: Aviation Safety Analyst
You are an ICAO-certified accident investigation expert generating realistic air crash report training data. Focus strictly on aviation accident scenarios.
Your task is to generate corresponding question-answer pairs based on my questions and the content of air crash investigation report I provide.
You need to return data strictly according to the data specification.

## Core Requirements:
1. Generate **4-8 reports** per batch
2. JSON structure must include:
   - "instruction": Specific analysis task
   - "input": Scenario parameters 
   - "output": Structured report following Annex 13 standards
3. Mandatory components:
   - IATA/ICAO aircraft codes
   - METAR weather data format
   - Flight phase terminology (e.g., rotation, FL350)
   - Human Factors Analysis Framework elements

## Scenario Matrix (Uniform coverage):
1. Accident type:
   - CFIT (Controlled Flight Into Terrain) 30%
   - Mid-air Collision 20%
   - System Failure 25%
   - Human Factors 25%
2. Stages distribution:
   - Pre-impact (Black Box Analysis) 40%
   - Post-crash (Wreckage investigation) 30%
   - Organizational Factors 30%
3. Complexity control:
   - Level 1: Single-factor analysis (e.g., pitot tube icing)
   - Level 2: Multi-system interaction 
   - Level 3: Crew resource management + ATC coordination

## Data specification:
```json
[
  {
    "instruction": "Analyze FDR data for stall warning activation sequence",
    "input": "Aircraft: B738, Flight Phase: Final Approach, Altitude: 1500ft AGL",
    "output": "FDR Parameter Analysis:\n- 23:45:11Z: STALL WARNING triggered (AOA 12.7°)\n- 23:45:12Z: IAS dropped to 124kt (Vref+5)..."
  },
  {
    "instruction": "Draft maintenance factor section for engine failure report",
    "input": "Event: Uncontained engine failure, Engine: CFM56-7B, Last Mx: 320 FH ago",
    "output": "Maintenance Review:\n1. Fan blade inspection records show...\n2. ESN 45122 had 2,983 cycles since last HSI...\nNTSB Recommendation: Implement ED-112 compliant..."
  }
]

## The contents of the air crash investigation report are as follows:
"""

{{air crash investigation report}}

"""
'''


class NTSBAviationReportsInstructionGenerator(object):
  """空难事故指令数据集生成器（从PDF文件）"""
  class Config(object):
    def __init__(self, conf_file_path:str):
      import yaml
      with open(conf_file_path, 'r') as file:
        config = yaml.safe_load(file)
      if 'proxy' in config['global']:
        self.proxy = config['global']['proxy']
      else:
        self.proxy = None
      self.api_key = os.environ.get('OPEN_AI_API_KEY')
      self.base_url = config['instruction_generator']['open_ai']['base_url']
      self.model = config['instruction_generator']['open_ai']['model']

  def __init__(self, pdf_dir:str, dataset_dir_path:str, conf_file_path:str='config.yaml'):
    self.config = NTSBAviationReportsInstructionGenerator.Config(conf_file_path)
    dataset_parent_dir = os.path.dirname(dataset_dir_path)
    if dataset_parent_dir != "":
      os.makedirs(dataset_parent_dir, exist_ok=True)
    os.makedirs(dataset_dir_path, exist_ok=True)
    self.dataset_dir_path = dataset_dir_path
    self._extract_text_from_pdf_dir(pdf_dir)

  def _extract_text_from_pdf_dir(self, pdf_dir:str) -> None:
    """提取pdf内容"""
    self.texts = []
    self.file_names = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=cpu_count() * 2) as t:
      tasks = []
      for filename in os.listdir(pdf_dir):
        def handler(file_path:str) -> None:
          doc = fitz.open(file_path)
          text = ""
          for page in doc:
            text += page.get_text()
          with lock:
            self.texts.append(text)
            self.file_names.append(filename)
        tasks.append(t.submit(handler, os.path.join(pdf_dir, filename)))
      wait(tasks)

  @staticmethod
  def qa_generate_by_single_report(config:Config, idx:int, cur_file_name:str, content:str, dataset_dir_path:str) -> None:
    from openai import OpenAI
    if config.proxy:
      os.environ["http_proxy"] = config.proxy['http']
      os.environ["https_proxy"] = config.proxy['https']
    client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def split_content(content, chunk_size=50000):
      return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

    content_chunks = split_content(content)
    all_answers = []

    for chunk in content_chunks:
      prompt = QUESTION_ANSWER_GENERATE_PROMPT_TEMPLATE.replace("{{air crash investigation report}}", chunk)
      try:
        response = client.chat.completions.create(
          model=config.model,
          messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "concat to question-answer pairs"}
          ],
          stream=False
        )
      except Exception as e:
        print(f"file {cur_file_name} generate failed: {e}")
        continue
      all_answers.append(response.choices[0].message.content)

    # 合并所有结果
    response = "\n".join(all_answers)
    with open(os.path.join(dataset_dir_path, str(idx) + ".json"), 'w', encoding='utf-8', errors="ignore") as f:
      f.write(NTSBAviationReportsInstructionGenerator.extract_json_from_response(response) + '\n')
    print(f"task {idx} finished")

  @staticmethod
  def extract_json_from_response(response:str) -> str:
    """从llm回答中提取出json"""
    result = response.split("```json\n")[1].split("```")[0]
    return result

  def _aggregate_jsons(self) -> None:
    import json
    import chardet
    result_file_path = os.path.join(self.dataset_dir_path, "all.json")
    if os.path.exists(result_file_path):
      os.remove(result_file_path)
    l = []
    for filename in os.listdir(self.dataset_dir_path):
      file_path = os.path.join(self.dataset_dir_path, filename)
      with open(file_path, 'rb') as f:
        raw_data = f.read()
        detected_encoding = chardet.detect(raw_data)['encoding']

      with open(file_path, "r", encoding=detected_encoding) as file:
        json_data = json.load(file)
        for x in json_data:
          l.append(x)
    with open(os.path.join(self.dataset_dir_path, "all.json"), "w", encoding="utf-8") as json_file:
      json.dump(l, json_file, ensure_ascii=False)

  def __call__(self) -> None:
    """生成数据集"""
    with ThreadPoolExecutor(max_workers=cpu_count() * 2) as t:
      tasks = []
      for index, content in enumerate(self.texts):
        handler = NTSBAviationReportsInstructionGenerator.qa_generate_by_single_report
        args = (self.config, index, self.file_names[index], content, self.dataset_dir_path)
        tasks.append(t.submit(lambda p: handler(*p), args))
      wait(tasks)
      print('all tasks done')
    # 聚合生成的多个json文件
    self._aggregate_jsons()

def parse_instruction_generator_arguments():
  parser = argparse.ArgumentParser(description="空难调查报告指令数据集生成器参数")
  parser.add_argument('--pdf_dir', type=str, default="reports", help='存放调查报告的文件夹')
  parser.add_argument('--dataset_dir_path', type=str, default="instructions", help='指令数据集产出目录')
  parser.add_argument('--conf_file_path', type=str, default="config.yaml", help='配置文件路径')
  return parser.parse_args()


if __name__ == '__main__':
  args = parse_instruction_generator_arguments()
  ac = NTSBAviationReportsInstructionGenerator(
    pdf_dir=args.pdf_dir,
    dataset_dir_path=args.dataset_dir_path,
    conf_file_path=args.conf_file_path
  )
  ac()
