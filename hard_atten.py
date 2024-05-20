import torch
import re
from data_loader import DBLPDataset
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt import *
import pandas as pd
import numpy as np
tqdm.pandas()
import time
import ast

def clean_ref_abstracts(data):
    if isinstance(data, str):  # Check if the data is a string
        try:
            # Convert string representation of list to actual list
            data = ast.literal_eval(data)
        except (ValueError, SyntaxError):
            # If the string can't be evaluated as a Python literal, return None or []
            return None
    # Further processing to replace 'nan' with None if data is now a list
    if isinstance(data, list):  # If the data is a list
        return [None if x == 'nan' else x for x in data]  # Replace 'nan' with None


def safe_eval(expr):
    if isinstance(expr, list):  # If it's a list, evaluate each element
        return [eval(x) if x is not None else None for x in expr]
    try:
        return eval(expr) if expr is not None else None
    except Exception as e:
        print(f"Error evaluating {expr}: {e}")
        return None

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

    # df = pd.read_csv('patent.csv')
    # df = df[['abstract', 'ref_abstracts', 'd']]
    # df['ref_abstract'] = df['ref_abstracts'].apply(clean_ref_abstracts)
    # # df['ref_abstract'] = df['ref_abstracts'].apply(safe_eval)
    # print(len(df))
    # print("Begin to generate hard attention!!!")
    # print(type(df['ref_abstract'].iloc[0]))
    # df_exploded = df.explode('ref_abstract').reset_index(drop=True)
    # print(df_exploded.head())
    # print(len(df_exploded))
    # df_exploded.dropna(inplace=True)
    df = pd.read_csv('patent_ex.csv')
    df_exploded = df[['d', 'abstract', 'ref_abstract']]
    df_exploded['input'] = df_exploded.apply(lambda row: patent_importance(row['abstract'], row['ref_abstract']), axis=1)
    df_exploded = df_exploded[['abstract', 'ref_abstract', 'd', 'input']].reset_index(drop=True)
    print(len(df_exploded))
    df_exploded['input_length'] = df_exploded['input'].apply(len)
    df_exploded_sorted = df_exploded.sort_values(by='input_length').reset_index(drop=True)
    print(len(df_exploded_sorted))
    # print(df_exploded_sorted.head(-5))
    print(df_exploded_sorted.head())
    # df_ex = df_exploded_sorted[['abstract', 'ref_abstract', 'd']]
    # df_ex.to_csv('patent_ex.csv')

    atten_dataset = DBLPDataset(df_exploded_sorted, tokenizer)
    print(atten_dataset)
    atten_loader = DataLoader(atten_dataset, batch_size=16, shuffle=False)
    atten_loader = tqdm(atten_loader)
    print('attention_down')

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    gen_kwargs = {"max_length": 8192, "num_beams": 1, "do_sample": True, "top_p": 0.8, "temperature": 0.8,
                  "logits_processor": None}

    results_df = pd.DataFrame()
    global_index = 0
    print('begin to batch')
    for batch in atten_loader:
        # print('11111111111111')
        time1 = time.time()
        input_ids = batch['input_ids'].to(device)
        # print(input_ids)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
        # print(outputs)
        decoded_outputs = []
        for i, output in enumerate(outputs):
            response = tokenizer.decode(output, skip_special_tokens=True)
            # print(response)
            pattern = r"Output:\s*(\d)"
            matches = re.findall(pattern, response)[1].strip() if re.findall(pattern, response) else 'nan'
            current_data = df_exploded_sorted.iloc[global_index][['d', 'abstract', 'ref_abstract']].copy()
            current_data['matches'] = matches
            decoded_outputs.append(current_data)
            global_index += 1

        batch_df = pd.DataFrame(decoded_outputs)
        results_df = pd.concat([results_df, batch_df], ignore_index=True)

        time2 = time.time()
        print(f"The time is {time2 - time1}s")
        results_df.to_csv('hard_attention_patent.csv', mode='a', header=False, sep='\t')
        results_df = pd.DataFrame()

        # python -u chatglm_train/model_glm/hard_atten.py --phase chatglm-32k