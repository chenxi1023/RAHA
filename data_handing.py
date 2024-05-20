import pandas as pd
import ast
from torch.utils.data import DataLoader
from data_loader import DBLPDataset
from prompt import prompt_generation

def create_data_loader(grouped_df, tokenizer, batch_size, shuffle=False):
    """Create DataLoader for given dataset."""
    dataset = DBLPDataset(grouped_df, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def difference_generation(differences):
    """Generate formatted prompts based on differences."""
    formatted_prompts = [f"No.{id + 1} difference: {diff}" for id, diff in enumerate(differences)]
    return ' '.join(formatted_prompts)

def update_prompts(df_new):
    """Update input prompts based on abstract and differences."""
    df_new['input'] = df_new.apply(lambda row: prompt_generation(row['abstract'], row['difference'], row.get('predictions', '')), axis=1)
    return df_new

def filter_df(row):
    """Filter data rows based on ast evaluations."""
    try:
        differences = ast.literal_eval(row['difference'])
    except (ValueError, SyntaxError):
        differences = row['difference']
    try:
        attens = ast.literal_eval(row['atten'])
    except (ValueError, SyntaxError):
        attens = row['atten']
    filtered_differences = [difference for difference, atten in zip(differences, attens) if atten == 1]
    filtered_attens = [1] * len(filtered_differences)
    return filtered_differences, filtered_attens

def filter_data(d):
    """Apply filters to data and update prompts."""
    d = d.copy()
    new_columns = d.apply(lambda row: filter_df(row), axis=1)
    d['difference_filtered'] = [item[0] for item in new_columns]
    d['atten_filtered'] = [item[1] for item in new_columns]
    d['difference_filtered'] = d['difference_filtered'].apply(lambda x: difference_generation(x))
    data = update_prompts(d)
    return data