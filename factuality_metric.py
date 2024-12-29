from base_metric import BaseMetric

class FactualityMetric(BaseMetric):
    """
    For demonstration: compares the model’s predicted label (or text) to a "reference" text or label.
    This is a placeholder for more advanced fact-checking approaches.
    """
    def __init__(self):
        super().__init__(name="factuality_metric")
    
    def compute(self, predicted_labels, reference_labels):
        """
        We'll define 'factuality' as simple accuracy for classification tasks.
        For a generative model, you'd do something more advanced (like BLEU, ROUGE, or exact string match).
        """
        correct = 0
        total = len(reference_labels)
        for pl, rl in zip(predicted_labels, reference_labels):
            if pl == rl:
                correct += 1
        return correct / total if total > 0 else 0
