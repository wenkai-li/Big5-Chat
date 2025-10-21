# step 1

# from huggingface_hub import snapshot_download

# snapshot_download(repo_id="allenai/soda", repo_type="dataset", cache_dir="/data/user_data/wenkail/llm_personality/soda_data/")

# step 2

import pandas as pd

splits = {'train': 'train.parquet', 'validation': 'valid.parquet', 'test': 'test.parquet'}

for split in ['train', 'validation', 'test']:
    df = pd.read_parquet("hf://datasets/allenai/soda/" + splits[split])
    df.to_csv(f"./soda_data/soda_{split}.csv")