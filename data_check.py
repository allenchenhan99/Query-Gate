import pandas as pd
import json

def check_data(file_path="dolly_processed.jsonl"):
    """
    check the data status of dolly_processed.jsonl
    """
    print(f"Reading {file_path}...")
    
    # read jsonl
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # DataFrame
    df = pd.DataFrame(data)
    
    print("\n" + "="*50)
    print("Data basic information")
    print("="*50)
    print(f"Total: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Columns names: {list(df.columns)}")
    
    print("\n" + "="*50)
    print("Category distribution")
    print("="*50)
    if 'category' in df.columns:
        category_counts = df['category'].value_counts().sort_index()
        print(f"Total {len(category_counts)} categories:")
        print()
        for category, count in category_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {category:20s}: {count:5d} rows ({percentage:5.2f}%)")
        print(f"\nTotal: {category_counts.sum()} rows")
    else:
        print("Warning: 'category' column not found")
    
    print("\n" + "="*50)
    print("Label distribution")
    print("="*50)
    if 'label' in df.columns:
        label_counts = df['label'].value_counts().sort_index()
        print(f"Total {len(label_counts)} labels:")
        print()
        for label, count in label_counts.items():
            label_name = "Fast Path (0)" if label == 0 else "Slow Path (1)"
            percentage = (count / len(df)) * 100
            print(f"  Label {label} ({label_name:15s}): {count:5d} rows ({percentage:5.2f}%)")
        print(f"\nTotal: {label_counts.sum()} rows")
    else:
        print("Warning: 'label' column not found")
    
    print("\n" + "="*50)
    print("Category and label relationship")
    print("="*50)
    if 'category' in df.columns and 'label' in df.columns:
        cross_tab = pd.crosstab(df['category'], df['label'], margins=True)
        print(cross_tab)


if __name__ == "__main__":
    check_data()
