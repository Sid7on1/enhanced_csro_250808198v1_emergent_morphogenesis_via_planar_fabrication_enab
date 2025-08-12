import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants and configuration
CONFIG = {
    'model_path': 'model.pth',
    'data_path': 'data.csv',
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001,
    'log_interval': 100
}

class BilayerSystemDataset(Dataset):
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data.iloc[idx]
        return {
            'input': torch.tensor(sample['input'].values, dtype=torch.float32),
            'output': torch.tensor(sample['output'].values, dtype=torch.float32)
        }

class BilayerSystemModel(nn.Module):
    def __init__(self):
        super(BilayerSystemModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AgentTrainer:
    def __init__(self, model: nn.Module, data_loader: DataLoader):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = Adam(self.model.parameters(), lr=CONFIG['learning_rate'])

    def train(self):
        for epoch in range(CONFIG['epochs']):
            for batch in self.data_loader:
                inputs, labels = batch
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = nn.MSELoss()(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if epoch % CONFIG['log_interval'] == 0:
                    logging.info(f'Epoch {epoch+1}, Loss: {loss.item()}')
        torch.save(self.model.state_dict(), CONFIG['model_path'])

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.data_loader:
                inputs, labels = batch
                outputs = self.model(inputs)
                loss = nn.MSELoss()(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(self.data_loader)

def main():
    # Load data
    dataset = BilayerSystemDataset(CONFIG['data_path'])
    data_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    # Create model
    model = BilayerSystemModel()

    # Train model
    trainer = AgentTrainer(model, data_loader)
    trainer.train()

    # Evaluate model
    loss = trainer.evaluate()
    logging.info(f'Test Loss: {loss}')

if __name__ == '__main__':
    main()