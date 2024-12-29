import torch
import torch.nn.functional as F
from base_metric import BaseMetric

# class ToxicityMetric(BaseMetric):
#     """
#     Checks how often the model's predicted label matches the reference label
#     for 'toxic' vs. 'non-toxic' classification.
#     """

#     def __init__(self):
#         super().__init__(name="toxicity_metric")

#     def compute(self, logits, reference_label):
#         # first index is "toxic" and second is "non-toxic."
#         # Argmax to get the predicted class.
#         probs = F.softmax(logits, dim=-1)
#         predicted_class = torch.argmax(probs, dim=-1).item()

#         # Reference_label to 0 if 'toxic', 1 if 'non_toxic'
#         ref_class = 0 if reference_label == "toxic" else 1

#         # Return 1 if correct, else 0
#         return 1 if predicted_class == ref_class else 0

class ToxicityMetric(BaseMetric):
    """
    Accuracy-based metric for binary classification: toxic (1) vs. non-toxic (0)
    """
    def __init__(self):
        super().__init__(name="toxicity_metric")

    def compute(self, logits_batch, true_labels):
        """
        logits_batch: (batch_size, num_classes)
        true_labels: list of 0/1 for the same batch
        """
        probs = F.softmax(logits_batch, dim=-1)
        predicted_classes = torch.argmax(probs, dim=-1).cpu().numpy()

        # Convert true labels to 0/1 if they are not already
        # For Jigsaw, it's often 0/1 out of the box for 'toxic'
        # But if it's a boolean or "toxic"/"non_toxic" string, we need to map.
        correct = 0
        total = len(true_labels)
        for pred, gold in zip(predicted_classes, true_labels):
            gold_int = gold if isinstance(gold, int) else int(gold)
            if pred == gold_int:
                correct += 1

        return correct, total
