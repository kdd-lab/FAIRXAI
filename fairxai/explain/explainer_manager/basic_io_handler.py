from abc import ABC, abstractmethod
from typing import List, Any


class BaseIoHandler(ABC):
    """
    Abstract base class for all wizard interaction handlers.

    Defines a standard interface for interactive framework usage.
    Subclasses (e.g., CLIHandler, GUIHandler, APIHandler) must implement these methods
    to handle user input/output according to their respective interaction mode.
    """

    @abstractmethod
    def ask_choice(self, title: str, options: List[Any], message: str) -> Any:
        """
        Displays a list of options to the user and returns the selected one.

        Args:
            title (str): Title of the current step or dialog.
            options (List[Any]): List of available options to choose from.
            message (str): Instruction or prompt message to the user.

        Returns:
            Any: The selected option.
        """
        pass

    @abstractmethod
    def show_message(self, message: str):
        """
        Displays a message, notification, or log entry to the user.

        Args:
            message (str): The message to display.
        """
        pass

    @abstractmethod
    def confirm(self, message: str) -> bool:
        """
        Asks the user for confirmation (yes/no) and returns their response.

        Args:
            message (str): The confirmation question.

        Returns:
            bool: True if the user confirms, False otherwise.
        """
        pass
