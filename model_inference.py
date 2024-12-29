import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class HFModelWrapper:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        """
        Returns logits or probabilities for a single text input. Return the raw logits.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        return logits
