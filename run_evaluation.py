import os
import torch
import csv
from torch.utils.data import  Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model_inference import HFModelWrapper
from toxicity_metric import ToxicityMetric
from data_loader import load_json

# def main():
#     # Load Data
#     data_file = os.path.join("Data", "test_data.json")
#     test_data = load_json(data_file)

#     # Initialize Model
#     model_wrapper = HFModelWrapper(model_name="distilbert-base-uncased-finetuned-sst-2-english")

#     # Initialize Metrics
#     toxicity_metric = ToxicityMetric()
#     total_samples = len(test_data)
#     correct_predictions = 0

#     # Run Evaluation
#     for item in test_data:
#         text = item["text"]
#         reference_label = item["label"]
#         logits = model_wrapper.predict(text)
        
#         # Compute metric
#         score = toxicity_metric.compute(logits, reference_label)
#         correct_predictions += score

#     # Summarize Results
#     accuracy = correct_predictions / total_samples * 100.0
#     print(f"Toxicity Accuracy: {accuracy:.2f}% on {total_samples} samples.")
    
#     # Save results in csv
#     with open('toxicity_results.csv', mode='w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow(["id", "text", "reference_label", "predicted_label", "correct"])
    
#         for item in test_data:
#             text = item["text"]
#             reference_label = item["label"]
#             logits = model_wrapper.predict(text)
#             score = toxicity_metric.compute(logits, reference_label)
#             predicted_class = torch.argmax(logits, dim=-1).item()
#             writer.writerow([item["id"], text, reference_label, predicted_class, score])


# if __name__ == "__main__":
#     main()

import os
import yaml
import csv
import argparse

from toxicity_metric import ToxicityMetric
from bias_metric import BiasMetric
from factuality_metric import FactualityMetric

from run_inference_utils import TextDataset, collate_fn, HFModelWrapper
from data_loader import load_jigsaw_dataset

from torch.utils.data import DataLoader

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = parse_config()

    # 1. Load Dataset
    data_list = load_jigsaw_dataset(config)
    
    # 2. Create Dataset / Dataloader
    ds = TextDataset(data_list, text_key="comment_text", label_key="toxic")
    dl = DataLoader(ds, batch_size=config["batch_size"], shuffle=False,
                    collate_fn=lambda x: collate_fn(x, HFModelWrapper(model_name=config["model_name"]).tokenizer))
    
    # 3. Initialize Model
    model_wrapper = HFModelWrapper(model_name=config["model_name"])
    
    # 4. Initialize Metrics
    metrics = []
    for m in config["metrics"]:
        if m == "toxicity":
            metrics.append(ToxicityMetric())
        elif m == "bias":
            metrics.append(BiasMetric())
        elif m == "factuality":
            metrics.append(FactualityMetric())
        else:
            print(f"Warning: Unknown metric {m}")

    # For storing results
    total_correct_toxic = 0
    total_samples = 0
    texts_for_bias = []
    predictions_for_bias = []
    ref_labels_for_fact = []
    pred_labels_for_fact = []

    # 5. Inference + Metric Computation
    # We'll compute toxicity in a single pass for demonstration.
    with open(config["output_path"], mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "gold_label", "predicted_label"])

        id_counter = 1
        for batch in dl:
            encodings, labels = batch
            logits = model_wrapper.predict_batch(encodings)
            
            # Evaluate Toxicity
            if any(isinstance(m, ToxicityMetric) for m in metrics):
                correct, batch_size = ToxicityMetric().compute(logits, labels)
                total_correct_toxic += correct
                total_samples += batch_size
            
            # Predicted Classes
            # (We reuse the logic from ToxicityMetric for a quick classification approach)
            predicted_classes = logits.argmax(dim=-1).cpu().numpy()
            
            # For Bias Metric
            if any(isinstance(m, BiasMetric) for m in metrics):
                # gather text + predicted label
                batch_texts = encodings["input_ids"]  # not the actual text, we need the original from ds...
                # For demonstration, let's just re-map the text from ds using an index
                # But DataLoader doesn't pass indexes by default. We'll do a simpler approach:
                # We'll store them separately (see below).
                pass
            
            # For Factuality Metric
            if any(isinstance(m, FactualityMetric) for m in metrics):
                # We'll store predicted and gold to compute factual metric
                pred_labels_for_fact.extend(predicted_classes)
                ref_labels_for_fact.extend(labels)

            # Write results to CSV
            # We can't get the raw text from the batch easily without indices.
            # Let's assume we store them in ds for demonstration:
            batch_size = len(labels)
            for i in range(batch_size):
                text_item = data_list[id_counter-1]["comment_text"]  # not robust for big data, but okay for demo
                gold_label = labels[i]
                pred_label = predicted_classes[i]
                writer.writerow([id_counter, text_item, gold_label, pred_label])
                id_counter += 1

                # For bias metric
                texts_for_bias.append(text_item)
                predictions_for_bias.append(pred_label)

    # 6. Summarize
    # Toxicity
    if any(isinstance(m, ToxicityMetric) for m in metrics):
        toxicity_acc = total_correct_toxic / total_samples if total_samples else 0
        print(f"[Toxicity] Accuracy: {toxicity_acc:.2f}")

    # Bias
    for m in metrics:
        if isinstance(m, BiasMetric):
            bias_results = m.compute(texts_for_bias, predictions_for_bias)
            print(f"[Bias] Results: {bias_results}")

    # Factuality
    for m in metrics:
        if isinstance(m, FactualityMetric):
            factual_score = m.compute(pred_labels_for_fact, ref_labels_for_fact)
            print(f"[Factuality] Score: {factual_score:.2f}")

if __name__ == "__main__":
    main()

