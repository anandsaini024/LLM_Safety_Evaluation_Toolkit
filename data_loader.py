import json
import random
from datasets import load_dataset

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_jigsaw_dataset(config):
    """
    Loads the Jigsaw Toxic Comment dataset from Hugging Face Datasets
    and returns a subset (config.subset_size).
    """
    dataset_name = config["dataset_name"]
    dataset_split = config["dataset_split"]
    subset_size = config["subset_size"]

    # Load the dataset
    ds = load_dataset(dataset_name, "raw", split=dataset_split)

    # For Jigsaw, columns might be: ["comment_text", "toxic", ...]
    # Let’s keep only "comment_text" and "toxic"
    ds = ds.remove_columns(["input_ids", "token_type_ids", "attention_mask"])

    # Convert to list of dict
    data_list = ds.to_dict("records")
    random.shuffle(data_list)
    
    # Subsample
    data_list = data_list[:subset_size]
    
    return data_list