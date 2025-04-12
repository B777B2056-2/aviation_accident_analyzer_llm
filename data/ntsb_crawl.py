#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import argparse
import os
import requests
from typing import List, Dict
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import multiprocessing
import threading
import uuid
import time
import random


class ReportDownloadThread(threading.Thread):
  """文件下载线程"""
  def __init__(self, save_path: str, links: List[str], proxies:Dict[str, str]=None):
    threading.Thread.__init__(self)
    self.save_path = save_path
    self.links = links
    self.proxies = proxies

  def run(self):
    for link in self.links:
      try:
        file_name = os.path.join(self.save_path, str(uuid.uuid4())+".pdf")
        with requests.get(link, proxies=self.proxies, stream=True) as r:
          r.raise_for_status()
          with open(file_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
              f.write(chunk)
        print(f"已下载: {file_name}")
      except Exception as e:
        print(f"下载失败 {link}: {e}")
      time.sleep(random.random() * 5)


class NTSBAviationReportsCrawler(object):
  """NTSB空难调查报告爬虫"""
  class Config(object):
    def __init__(self, conf_file_path:str):
      import yaml
      with open(conf_file_path, 'r') as file:
        config = yaml.safe_load(file)
      if 'proxy' in config['global']:
        self.proxy = config['global']['proxy']
      else:
        self.proxy = None
      self.max_html_page_num = config['ntsb_crawl']['max_html_page_num']

  def __init__(self, save_dir:str, conf_file_path:str):
    self.config = NTSBAviationReportsCrawler.Config(conf_file_path)
    self.save_dir = save_dir
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    self.url = "https://www.ntsb.gov/investigations/AccidentReports/Pages/Reports.aspx"
    parsed_url = urlparse(self.url)
    self.base_url = parsed_url.scheme + "://" + parsed_url.netloc

  def _do_crawl(self):
    """爬取网页"""
    browser = webdriver.Edge()
    browser.get(self.url)

    # 选择Aviation
    button = browser.find_element(By.ID, 'btnaviation')
    button.click()
    time.sleep(random.random() * 5)

    self.download_links = []
    for page in range(self.config.max_html_page_num):
      # 获取html内容
      soup = BeautifulSoup(browser.page_source, 'html.parser')
      for div_tag in soup.find_all("div", attrs={'class': 'download'}):
        url_tag = div_tag.find_all("a")[0]
        print(url_tag.get("href"))
        self.download_links.append(self.base_url + url_tag.get("href"))
      time.sleep(random.random() * 5)
      # 翻页
      element = browser.find_element(By.ID, 'NextLinkButton')
      browser.execute_script("arguments[0].click();", element)
    browser.quit()

  def _split_threads(self):
    """切分线程和文件"""
    threads = []
    threads_num = multiprocessing.cpu_count()
    single_thread_link_num = int(len(self.download_links) // threads_num)
    last_thread_link_num = len(self.download_links) % threads_num
    # 无需多线程，单线程直接下载
    if single_thread_link_num == 0:
      threads.append(
        ReportDownloadThread(save_path=self.save_dir, links=self.download_links, proxies=self.config.proxy)
      )
      return threads
    # 多线程下载
    if last_thread_link_num != 0:
      threads_num += 1
    print(f"file count: {len(self.download_links)}")
    for i in range(threads_num):
      start = i * single_thread_link_num
      if i != threads_num - 1:
        end = (i + 1) * single_thread_link_num
      else:
        end = -1
      threads.append(
        ReportDownloadThread(save_path=self.save_dir, links=self.download_links[start:end], proxies=self.config.proxy)
      )
    return threads

  def _download_reports(self):
    """文件下载"""
    threads = self._split_threads()
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()

  def __call__(self):
    print("do crawl...")
    self._do_crawl()
    print("downloading...")
    self._download_reports()
    print("done")


def parse_crawl_arguments():
    parser = argparse.ArgumentParser(description="NTSB调查报告爬虫参数")
    parser.add_argument('--save_dir', type=str, default="reports", help='调查报告保存路径')
    parser.add_argument('--conf_file_path', type=str, default="config.yaml", help='配置文件路径')
    return parser.parse_args()


if __name__ == "__main__":
  args = parse_crawl_arguments()
  crawler = NTSBAviationReportsCrawler(save_dir=args.save_dir, conf_file_path=args.conf_file_path)
  crawler()
