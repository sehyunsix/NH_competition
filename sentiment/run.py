import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys
import click
from datasets import load_dataset
from datasets import Dataset, ClassLabel, Value, Features



class Bert():
    def __init__(self,model,dataset,subset,split,batchsize,output_dir):
        self.device = torch.device('cuda')
        self.model = BertForSequenceClassification.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if subset ==None :
            self.dataset = load_dataset(dataset,subset,split=split)
        else:
            self.dataset = load_dataset(dataset,subset,split=split)
        self.headline_inputs = self.encoding(self.dataset['headline'])
        self.summary_inputs = self.encoding(self.dataset['summary'])
        self.batchsize = batchsize
        self.path = output_dir

    def encoding(self,texts):
        print('texts are endcoding now ....')
        output = self.tokenizer(texts, padding=True ,truncation=True,return_tensors="pt")
        results = output['input_ids']
        return results

    def inference(self,inputs):
        print("inferece_start")
        model = self.model
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        inputs.to(self.device)
        results =[]
        chunk_size= self.batchsize
        for i in tqdm(range(0, len(inputs), chunk_size)):
            chunk = inputs[i:i+chunk_size]
            chunk.to(self.device)
            # Perform inference on the chunk
            with torch.no_grad():
                output = model(chunk).logits
            results.append(output)
        outputs = torch.cat(results, dim=0)
        outputs = outputs.to('cpu').tolist()
        outputs = [{'postive':x[0],'negative':x[1],'neutral':x[2]} for x in outputs]
        return outputs
# Concatenate or process the results as needed

@click.command()
@click.option('--dataset', default='sehyunsix/Finnhub-News', help='dataset_name.')
@click.option('--subset', default ='clean',
              help='The person to greet.')
@click.option('--split', default='clean',
              help='split name.')
@click.option('--model', default='model',
              help='model name')
@click.option('--batchsize', default= 1024,
              help='batchsize number')
@click.option('--output_dir', default='data/clean_data',
              help='output_dir number')

def main(
    dataset: str,
    subset: str,
    output_dir: str,
    model: str,
    batchsize: int,
    split: str,
):
    bert = Bert(model,dataset,subset,split,batchsize,output_dir)
    headline_outputs =bert.inference(bert.headline_inputs)
    summary_outputs = bert.inference(bert.summary_inputs)
    data = {
        'headline': bert.dataset['headline'],
        'summary':bert.dataset['summary'],
        'headline_sentiment': headline_outputs,
        'summary_sentiment':summary_outputs
    }
    features =Features({
        'headline': Value('string'),
        'summary':Value('string'),
        'headline_sentiment': Value('dict'),
        'summary_sentiment':Value('dict')
    })
    custom_dataset = Dataset.from_dict(data, features=features)

if __name__ == '__main__':
    main()

