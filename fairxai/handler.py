from abc import abstractmethod

class Handler():
    """
    Abstract class for xai methodology handler
    """
    @abstractmethod
    def compatible_dataset(self):
        """
        it access object properties and return the compatible dataset types for the xai method handled: Tabular | Image | Text | Timeseries
        """

    def explanation_type(self):
        """
        it access object properties and return  the explanation types created by the xai method handled:
        Feature Importance | Rules-CounterRules | Examples-CounterExamples
        """