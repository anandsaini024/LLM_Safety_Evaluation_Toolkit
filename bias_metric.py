import numpy as np
from base_metric import BaseMetric

class BiasMetric(BaseMetric):
    """
    Placeholder for a bias metric: 
    For example, we can check if there's a difference in toxicity predictions 
    when certain demographic keywords appear in the text (e.g., "women", "men", "Black", "White", "gay", "straight").
    """
    def __init__(self, demographic_terms=None):
        super().__init__(name="bias_metric")
        if demographic_terms is None:
            # Simple list for demonstration
            self.demographic_terms = ["women", "men", "black", "white", "gay", "straight"]
        else:
            self.demographic_terms = demographic_terms
    
    def compute(self, texts, predictions):
        """
        We'll do a naive approach: if a text contains a demographic term, note the predicted class. 
        Then check if there's a difference in the rate of 'toxic' among different terms.

        predictions: list of 0/1 (0=non-toxic, 1=toxic)
        """
        results = {}
        for term in self.demographic_terms:
            indices = [i for i, txt in enumerate(texts) if term in txt.lower()]
            if len(indices) == 0:
                continue
            # gather predictions
            relevant_preds = [predictions[i] for i in indices]
            avg_toxic = np.mean(relevant_preds)
            results[term] = avg_toxic
        
        return results  # e.g. {"women": 0.4, "black": 0.7, ...}
