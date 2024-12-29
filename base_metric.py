class BaseMetric:
    """
    Abstract class for evaluating a model's output with a specific metric.
    """
    def __init__(self, name):
        self.name = name

    def compute(self, model_output, reference):
        """
        Compute the metric given model outputs and reference (gold labels or expectations).
        """
        raise NotImplementedError("Subclasses must implement compute method.")
