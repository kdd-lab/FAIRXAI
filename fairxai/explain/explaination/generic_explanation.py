class GenericExplanation:
    """
    Represents a generic explanation container for managing, summarizing, and visualizing explanation data.
    This class is designed to store and manipulate explanation data, including metadata about explainer
    name and type of explanation. It provides methods to convert the object to a dictionary representation,
    generate a summary of its content, and define placeholder functionality for visualization to be
    implemented by subclasses.
    Attributes
    ----------
    explainer_name : str
        The name of the explainer providing the explanation.
    explanation_type : str
        The type of explanation provided. Options are "local" or "global".
    data : dict
        A dictionary containing the explanation data to be processed.
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
        Converts the object instance data into a dictionary format.
        Returns
        -------
        dict
            A dictionary containing the object's `explainer` name, `type` of
            explanation, and associated `data` values.
        """
        # Convert object to dictionary for serialization or data transfer
        return {
            "explainer": self.explainer_name,
            "type": self.explanation_type,
            "data": self.data
        }

    def summarize(self) -> str:
        """
        Generates a detailed summary by compiling information stored in the object's attributes.
        The summary includes the explainer name, explanation type, and keys with their corresponding values
        from the data dictionary. For longer values, only the first 100 characters are included in the summary.
        Returns
        -------
        str
            A formatted summary string encapsulating object details.
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
        Formats a value for display in the summary, truncating if necessary.

        Parameters
        ----------
        value : any
            The value to format.

        Returns
        -------
        str
            The formatted value string, truncated if longer than MAX_VALUE_LENGTH.
        """
        # Convert value to string
        value_str = str(value)

        # Truncate if longer than MAX_VALUE_LENGTH (100 chars)
        if len(value_str) > self.MAX_VALUE_LENGTH:
            return f"{value_str[:self.MAX_VALUE_LENGTH]}..."
        return value_str

    def visualize(self):
        """
        Provides the visualization logic for implementing custom graphical or
        other forms of representations based on the context. This method is
        intended to be overridden by subclasses to define specific behavior.

        Raises
        ------
        NotImplementedError
            If the method is not overridden in a subclass.
        """
        # Abstract method - must be implemented by subclasses
        # Template Method Pattern: define the interface, delegate implementation
        raise NotImplementedError(
            f"visualize() must be implemented by subclasses of {self.__class__.__name__}"
        )
