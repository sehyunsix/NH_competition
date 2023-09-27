import pandas as pd
import sys
sys.path.append("/home/ssu36/tiger/NH_competition/FinNLP")
from finnlp.data_sources.news.finnhub_date_range import Finnhub_Date_Range
from finnlp.data_sources.news.yahoo_streaming import Yahoo_Date_Range

STK_QUT = pd.read_csv('STK_QUT.csv')
STK_IEM = pd.read_csv('STK_IEM.csv',encoding='EUC-KR')

data = list(STK_IEM['tck_iem_cd'].unique())

#name data
name_data = list(STK_IEM['fc_sec_eng_nm'].unique())
named_data = [ x.split() for x in name_data]
name_data = [ " ".join(x) for x in named_data]

#ticker data
ticker_data = [ x.split()[0] for x in data]


ticker={}
argus ={}
argus['token'] = 'ck5vse9r01qls0umds50ck5vse9r01qls0umds5g'
ticker_sample = [ ticker_data[x] for x in range(419,2743)]

import jsonlines
from pathlib import Path
from tqdm import tqdm
start_date = "2023-01-01"
end_date = "2023-08-30"

for ticker_name in tqdm(ticker_sample):
  news_downloader = Yahoo_Date_Range(argus)
  news_downloader.download_date_range_stock(start_date, end_date, ticker_name)
  df = news_downloader.dataframe
  df = df.dropna()
  df['datetime'] = df['datetime'].apply(lambda x : x.strftime('%Y-%m-%d'))
  json_lines = "\n".join( df.apply(lambda x: x.to_json(force_ascii= False), axis=1))
  with open( Path('/home/ssu36/tiger/NH_competition/data', 'train', f'{ticker_name}.jsonl'), "w"
          ) as fout:
            fout.write(json_lines)
