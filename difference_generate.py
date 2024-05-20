import torch
import re
from data_loader import DBLPDataset
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt import *
import pandas as pd
tqdm.pandas()
import numpy as np
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GLM model for generating differences.")
    parser.add_argument('--phase', type=str, help="Phase to control which model to use", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.phase == 'chatglm':
        tokenizer = AutoTokenizer.from_pretrained("chatglm3-6b", trust_remote_code=True)
        model_name = "chatglm3-6b"
    elif args.phase == 'chatglm-32k':
        tokenizer = AutoTokenizer.from_pretrained("chatglm3-6b-32k", trust_remote_code=True)
        model_name = "chatglm3-6b-32k"
    elif args.phase == 'chatglm-base':
        tokenizer = AutoTokenizer.from_pretrained("chatglm3-6b-base", trust_remote_code=True)
        model_name = "chatglm3-6b-base"
    else:
        raise ValueError("Unsupported phase argument")

    # df = pd.read_csv('data/dblp_all.csv', sep='\t', header=None,
    #                  names=['index', 'd', 'abstract', 'reference', 'atten'])
    data = pd.read_csv(f'data/dblp_all.csv', sep='\t', header=None,
                     names=['index', 'd', 'abstract', 'reference', 'atten', 'difference'])
    # df = pd.read_csv('data/dblp_all.csv.csv')
    df = data[data['atten'] == 0]
    print(df)
    df_exploded = df[['d', 'abstract', 'reference']]
    # print(df_exploded)
    print("Begin to generate difference!!!")
    df_exploded['input'] = df_exploded.apply(lambda row: prompt_difference(row['abstract'], row['reference']), axis=1)
    df_exploded = df_exploded[['abstract', 'reference', 'd', 'input']].reset_index(drop=True)
    print(len(df_exploded))
    df_exploded['input_length'] = df_exploded['input'].apply(len)
    df_exploded_sorted = df_exploded.sort_values(by='input_length', ascending=False).reset_index(drop=True)
    print(len(df_exploded_sorted))
    print(df_exploded_sorted.head(-5))
    print(df_exploded_sorted.head())

    atten_dataset = DBLPDataset(df_exploded_sorted, tokenizer)
    atten_loader = DataLoader(atten_dataset, batch_size=16, shuffle=False)
    atten_loader = tqdm(atten_loader)

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    gen_kwargs = {"max_length": 8192, "num_beams": 1, "do_sample": True, "top_p": 0.8, "temperature": 0.8,
                  "logits_processor": None}

    results_df = pd.DataFrame()
    global_index = 0
    for batch in atten_loader:
        time1 = time.time()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

        decoded_outputs = []
        for i, output in enumerate(outputs):
            response = tokenizer.decode(output, skip_special_tokens=True)
            # print(response)
            pattern = r"Contrast and Difference:\s*(.*?)(?=\n|$)"
            matches = re.findall(pattern, response)[0].strip() if re.findall(pattern, response) else 'nan'
            current_data = df_exploded_sorted.iloc[global_index][['d', 'abstract', 'reference']].copy()
            current_data['matches'] = matches
            decoded_outputs.append(current_data)
            global_index += 1

        batch_df = pd.DataFrame(decoded_outputs)
        results_df = pd.concat([results_df, batch_df], ignore_index=True)

        time2 = time.time()
        print(f"The time is {time2 - time1}s")
        results_df.to_csv('dblp_atten_0_sort.csv', mode='a', header=False, sep='\t')
        results_df = pd.DataFrame()