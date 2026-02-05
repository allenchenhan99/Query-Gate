from datasets import load_dataset
import pandas as pd
import os

def download_and_process_data():
    print("Downloading databricks-dolly-15k from Hugging Face...")
    
    # [cite_start]1. use the official package to download the data [cite: 7]
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    # convert to Pandas DataFrame for better handling
    df = pd.DataFrame(dataset)
    
    print(f"Original data size: {len(df)}")
    
    # [cite_start]2. filter the categories according to the requirements [cite: 8]
    # Fast Path (0): classification, summarization
    # Slow Path (1): creative_writing, general_qa
    target_categories = [
        'classification', 
        'summarization', 
        'creative_writing', 
        'general_qa'
    ]
    
    # only keep these four categories
    df_filtered = df[df['category'].isin(target_categories)].copy()
    
    # 3. add labels
    # Fast Path (label=0)
    df_filtered.loc[df_filtered['category'].isin(['classification', 'summarization']), 'label'] = 0
    # Slow Path (label=1)
    df_filtered.loc[df_filtered['category'].isin(['creative_writing', 'general_qa']), 'label'] = 1
    
    # convert to integers
    df_filtered['label'] = df_filtered['label'].astype(int)
    
    print(f"Filtered data size: {len(df_filtered)}") # the problem said it should be around 6224 [cite: 8]
    
    # jsonl
    output_file = "dolly_processed.jsonl"
    df_filtered.to_json(output_file, orient="records", lines=True)
    
    print(f"Processing completed! Data saved to {output_file}")

if __name__ == "__main__":
    download_and_process_data()