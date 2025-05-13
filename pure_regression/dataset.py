import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler

class VideoViewsDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
        # Scale numeric features
        self.scaler = StandardScaler()
        numeric_features = ['year', 'month', 'day', 'day_of_week', 'quarter', 'days_since_upload']
        self.df_numeric = pd.DataFrame(self.scaler.fit_transform(df[numeric_features]), 
                                      columns=numeric_features)
        
        # Get categorical features
        self.channel_ids = torch.tensor(df['channel_encoded'].values, dtype=torch.long)
        
        # Log transform the target
        self.log_views = np.log1p(df['views'].values)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get numeric features
        numeric_features = torch.tensor(self.df_numeric.iloc[idx].values, dtype=torch.float32)
        
        # Get categorical features
        channel_id = self.channel_ids[idx]
        
        # Get target
        log_views = torch.tensor(self.log_views[idx], dtype=torch.float32)
        
        return channel_id, numeric_features, log_views