class GenericExplanation:
    """
    A container class for managing and summarizing explanatory data.

    This class serves as a base for handling explanation data provided by different
    explainers. It allows the storage, serialization, summarization, and visualization
    of explanatory information. Explanation types can either be local or global, as defined
    by the constants. Subclasses can extend its functionality, particularly by implementing
    the visualization logic.
    """

    # Constants for explanation types
    LOCAL_EXPLANATION = "local"
    GLOBAL_EXPLANATION = "global"

    # Constant for maximum value length in summary
    MAX_VALUE_LENGTH = 100

    def __init__(self, explainer_name: str, explanation_type: str, data: dict):
        # Initialize the explanation container with name, type, and data
        self.explainer_name = explainer_name
        self.explanation_type = explanation_type  # "local" | "global"
        self.data = data

    def to_dict(self) -> dict:
        """
        Convert the object into a dictionary representation.

        This method serializes the object to a dictionary format suitable
        for data transfer or storage. It includes essential properties
        like the explanation type, the explainer name, and the data content.

        Returns:
            dict: A dictionary containing the object's data as key-value pairs.
        """
        # Convert object to dictionary for serialization or data transfer
        return {
            "explainer": self.explainer_name,
            "type": self.explanation_type,
            "data": self.data
        }

    def summarize(self) -> str:
        """
        Generate a formatted summary of explanation details.

        This method combines details about the explainer's name, type, and additional
        data into a readable, string-based summary. The data items are iterated over
        and their values are formatted for better presentation.

        Returns:
            str: A string summarizing the explainer's name, type, and data items.
        """
        # Create header with explainer name and type
        summary = f"Explanation {self.explainer_name} ({self.explanation_type})\n"

        # Iterate through all data items and format them
        for key, value in self.data.items():
            formatted_value = self._format_value(value)
            summary += f" - {key}: {formatted_value}\n"
        return summary

    def _format_value(self, value) -> str:
        """
        Converts a given value to its string representation and truncates it to a maximum length
        if it exceeds the predefined limit.

        Parameters:
        value: Any
            The value to be converted to a string representation.

        Returns:
        str
            The string representation of the value, truncated if it exceeds the maximum allowed length.
        """
        # Convert value to string
        value_str = str(value)

        # Truncate if longer than MAX_VALUE_LENGTH (100 chars)
        if len(value_str) > self.MAX_VALUE_LENGTH:
            return f"{value_str[:self.MAX_VALUE_LENGTH]}..."
        return value_str

    def visualize(self):
        """
        Abstract method to be implemented by subclasses for visualization.

        This method serves as a template method that subclasses must override to
        provide their specific implementation for visualization logic. It is
        intentionally left unimplemented in the base class and raises a
        NotImplementedError if called directly.

        Raises:
            NotImplementedError: Always raised when the method is invoked, indicating
            that it must be implemented by a subclass.
        """
        # Abstract method - must be implemented by subclasses
        # Template Method Pattern: define the interface, delegate implementation
        raise NotImplementedError(
            f"visualize() must be implemented by subclasses of {self.__class__.__name__}"
        )
