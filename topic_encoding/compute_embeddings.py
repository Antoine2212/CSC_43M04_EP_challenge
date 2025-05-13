from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

df = pd.read_csv('../dataset/train_val.csv')

sentence_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print("Pre-computing embeddings...")
all_embeddings = []
all_targets = []

for idx in tqdm(range(len(df))):
    row = df.iloc[idx]
    # Get embeddings using your existing method
    title = row['title']
    description = row['description']
    channel = row['channel']
    date = row['date']
    target = row['views']
    field_embeddings = sentence_encoder.encode([title, description, channel, date], convert_to_tensor=True)

    # Store them in numpy arrays
    all_embeddings.append(field_embeddings.cpu().numpy())
    view_count = row['views']
    all_targets.append(np.log1p(view_count))
    
# normalize targets
all_targets = np.array(all_targets)
all_targets = (all_targets - np.mean(all_targets)) / np.std(all_targets)
all_embeddings = np.array(all_embeddings)
print("Embeddings pre-computed.")

df['embeddings'] = all_embeddings.tolist()
df['targets'] = all_targets.tolist()
df.to_csv('../dataset/train_val_embeddings.csv', index=False)
print("Embeddings saved to CSV.")