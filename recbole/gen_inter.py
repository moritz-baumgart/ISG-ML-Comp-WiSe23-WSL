from mlcomp.data.load import load_regression_train
from pathlib import Path


df = load_regression_train()
df = df.drop_duplicates()

df = df.rename(columns={'item': 'item:token', 'user': 'user:token', 'timestamp': 'timestamp:float', 'rating': 'rating:float'})
Path('datasets/reg-train/').mkdir(parents=True, exist_ok=True)
df.to_csv('datasets/reg-train/reg-train.inter', sep='\t', index=False)