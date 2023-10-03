from datasets import load_dataset
import json
import os


## path =['clean_data','sentiment_data']
def save_to_json(df,name,path):
  file_path =f'../../data/{path}/'
  json_data = df.to_json(orient='records')
  with open(file_path, 'w') as json_file:
      json_file.write(json_data)

def upload_hub(name):
  ds = load_dataset(
        "json",
        data_files={
            "data":f"/home/ssu36/tiger/NH_competition/data/{name}/*.json"
        },
    )
  ds.push_to_hub('c',name)

def merge_jsonl_files(directory_path, output_file):
    # List all files in the directory
    files = os.listdir(directory_path)

    # Filter JSONL files
    jsonl_files = [file for file in files if file.endswith('.jsonl')]

    # Open the output file in write mode
    with open(output_file, "w", encoding="utf-8") as merged_file:
        for jsonl_file in jsonl_files:
            # Open each JSONL file in read mode
            with open(os.path.join(directory_path, jsonl_file), "r", encoding="utf-8") as file:
                for line in file:
                    # Read each line from the JSONL file and write it to the merged file
                    merged_file.write(line)
