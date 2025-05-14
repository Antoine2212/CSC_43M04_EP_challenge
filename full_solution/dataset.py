import torch
from PIL import Image
from torch.utils.data import Dataset

class MultiModalDataset(Dataset):
    def __init__(self, df, image_transform=None, image_dir="../dataset/train_val"):
        self.df = df
        self.image_transform = image_transform
        self.image_dir = image_dir

        # Scale numeric features
        self.numeric_features = ['year', 'month', 'day', 'day_of_week', 'quarter', 'days_since_upload']
        self.df_numeric = df[self.numeric_features].values

        # Get categorical features
        self.channel_ids = df['channel_encoded'].values

        # Get topic embeddings
        self.topic_embeddings = df['embeddings'].values

        # Log transform the target
        self.views = df['views'].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Get numeric features
        numeric_features = torch.tensor(self.df_numeric[idx], dtype=torch.float32)
        # Get categorical features
        channel_id = torch.tensor(self.channel_ids[idx], dtype=torch.long)
        # Get image features
        image_features = Image.open(f"{self.image_dir}/{row['id']}.jpg").convert("RGB")
        image_features = self.transform(image_features)
        # Get topic embeddings
        topic_embeddings = torch.tensor(self.topic_embeddings[idx], dtype=torch.float32)
        # Get views
        views = torch.tensor(self.views[idx], dtype=torch.float32)
        
        return channel_id, numeric_features, image_features, topic_embeddings, views
        
        
        
