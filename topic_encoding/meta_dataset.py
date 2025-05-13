import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer


class MetadataDataset(Dataset):
    def __init__(self, df):
        self.df = df
          
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        field_embeddings = row['embeddings']
        target = row['targets']
        
        return torch.tensor(field_embeddings, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)