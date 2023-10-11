from top2vec import Top2Vec

from datasets import load_dataset ,load_from_disk
import pandas as pd
from tqdm import tqdm

dataset_path =[]
dataset_list= []
sentiment_model_names = ['distilroberta','Bert+','deberta','finBert']
mrm_dataset = load_from_disk('data/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis')
ahm_dataset = load_from_disk('data/clean_data/ahmedrachid/FinancialBERT-Sentiment-Analysis')
nick_dataset= load_from_disk('data/nickmuchi/deberta-v3-base-finetuned-finance-text-classification')
pros_dataset= load_from_disk('data/ProsusAI/finbert')
dataset_date =load_dataset('sehyun66/Finnhub-News','clean',split='clean')

mrm_dataframe = pd.DataFrame(mrm_dataset)
ahm_dataframe = pd.DataFrame(ahm_dataset)
nick_dataframe = pd.DataFrame(nick_dataset)
pros_dataframe = pd.DataFrame(pros_dataset)
dataset_date = pd.DataFrame(dataset_date)


sentiment_df_list =[]
sentiment_df_list.append(mrm_dataframe)
sentiment_df_list.append(ahm_dataframe)
sentiment_df_list.append(nick_dataframe)
sentiment_df_list.append(pros_dataframe)

for idx,df in enumerate(sentiment_df_list):
  df['headline_label'] = df['headline_sentiment'].apply(lambda x: max(x,key=x.get))
  df['summary_label']= df['summary_sentiment'].apply(lambda x: max(x,key=x.get))
  df['negative_sentiment']= df['headline_sentiment'].apply(lambda x: float(x['negative']))


document_list=[]
for df in sentiment_df_list:
  negative_documents_headline =list(df[df['headline_label']=='negative']['headline'])
  postive_documents_headline =list(df[df['headline_label']=='postive']['headline'])
  negative_documents_summary =list(df[df['headline_label']=='negative']['summary'])
  postive_documents_summary =list(df[df['headline_label']=='postive']['summary'])
  diction ={}
  diction['negative_headline'] = negative_documents_headline
  diction['postive_headline'] = postive_documents_headline
  diction['negative_summary']  = negative_documents_summary
  diction['postive_summary'] = postive_documents_summary
  document_list.append(diction)

for i,document in tqdm(enumerate(document_list)):
  for item in tqdm(document.items()):
    model = Top2Vec(item[1])
    model.save(f"data_{i}/{item[0]}")
