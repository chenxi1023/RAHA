import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import argparse
import pandas as pd

from data_handling import create_data_loader, filter_data
from model.chatglm_adapter import chatglm_adapter
from model.chatglm_mlp import chatglm_mlp
from model.tea import chatglm_tea

def train_model(model, data_loader, optimizer, device, epochs):
    """Function to train the model."""
    model.train()
    for epoch in range(epochs):
        for batch in tqdm(data_loader, desc=f"Training Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask).squeeze(-1)
            loss = nn.MSELoss()(predictions, labels.float())
            loss.backward()
            optimizer.step()

def validate_model(model, data_loader, device):
    """Function to validate the model."""
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            predictions = model(input_ids, attention_mask).squeeze(-1)
            loss = nn.MSELoss()(predictions, labels.float())
            total_val_loss += loss.item()
    return total_val_loss / len(data_loader)

def test_model(model, data_loader, device):
    """Function to test the model."""
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            predictions = model(input_ids, attention_mask).squeeze(-1)
            loss = nn.MSELoss()(predictions, labels.float())
            total_test_loss += loss.item()
    return total_test_loss / len(data_loader)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.adapter:
        model = chatglm_adapter(args.model).to(device)
    elif args.att:
        model = chatglm_tea(args.model).to(device)
    else:
        model = chatglm_mlp(args.model).to(device)

    data = pd.read_csv(args.data)
    filtered_data = filter_data(data)
    train_data_loader = create_data_loader(filtered_data, tokenizer, args.batch_size, shuffle=True)
    val_data_loader = create_data_loader(filtered_data, tokenizer, args.batch_size)  # Assuming same data for simplification
    test_data_loader = create_data_loader(filtered_data, tokenizer, args.batch_size)  # Assuming same data for simplification

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if args.phase == 'train':
        train_model(model, train_data_loader, optimizer, device, args.epochs)
        val_loss = validate_model(model, val_data_loader, device)
        print(f"Validation Loss: {val_loss}")
    elif args.phase == 'validate':
        val_loss = validate_model(model, val_data_loader, device)
        print(f"Validation Loss: {val_loss}")
    elif args.phase == 'test':
        test_loss = test_model(model, test_data_loader, device)
        print(f"Test Loss: {test_loss}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GLM model for generating summary.")
    parser.add_argument('--phase', choices=['train', 'validate', 'test'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', default='data.csv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    main(args)