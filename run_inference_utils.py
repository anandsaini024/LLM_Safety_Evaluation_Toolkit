import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TextDataset(Dataset):
    def __init__(self, data_list, text_key="comment_clean", label_key="labels"):
        self.data_list = data_list
        self.text_key = text_key
        self.label_key = label_key

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        return {
            "text": item[self.text_key],
            "label": item[self.label_key]
        }

def collate_fn(batch, tokenizer):
    texts = [b["text"] for b in batch]
    labels = [b["label"] for b in batch]

    encodings = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    return encodings, labels

class HFModelWrapper:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, encodings):
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = self.model(**encodings)
        # Return logits
        return outputs.logits
