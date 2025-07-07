from abc import abstractmethod




class Handler():
    """
    Abstract class for xai methodology handler
    """
    @abstractmethod
    def set_compatible_dataset(self):
        """
        it sets the compatible dataset types for the xai method handled: 
        Feature Importance | Rules-CounterRules | Examples-CounterExamples
        """

    def set_explanation_type(self):
        """
        it sets the explanation types created by the xai method handled
        """