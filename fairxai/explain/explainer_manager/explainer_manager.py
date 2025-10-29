from typing import List, Type, Any

from fairxai.data.dataset.dataset_factory import DatasetFactory
from fairxai.explain.adapter.generic_explainer_adapter import GenericExplainerAdapter
from fairxai.logger import logger


class SingletonExplainerManager(type):
    """
    Metaclass that implements the Singleton design pattern.

    This metaclass ensures that a class using it can only have one instance. If an
    instance exists, it returns the existing instance; otherwise, it creates a new
    one.

    Attributes:
        _instances (dict): Dictionary to store the single instances of classes
            using this metaclass.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Handles the creation of singleton instances for classes. Ensures that only one
        instance of a class exists and provides a global point of access to it.

        Parameters:
            cls: The class for which the instance needs to be created.
            *args: Positional arguments to be passed to the class constructor.
            **kwargs: Keyword arguments to be passed to the class constructor.

        Returns:
            The singleton instance of the class.

        Raises:
            Does not explicitly raise any error but can propagate exceptions raised
            by the class constructor.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class ExplainerManager(metaclass=SingletonExplainerManager):
    """
    Manages the registration, initialization, and use of explainer classes.

    This class serves as a central registry for managing explainer classes that
    extend the `GenericExplainerAdapter`. It provides functionality for registering
    new explainer classes, retrieving and initializing them, as well as managing
    the generation of explanations for individual instances or global analyses.

    Additionally, this class ensures compatibility between explainers, models, and
    datasets, making it a robust framework for handling various explanation needs.
    """

    def __init__(self):
        """
        Initializes the attributes required for the class to store and manage explainers.

        Attributes:
            explainers (dict[str, Type[GenericExplainerAdapter]]): A dictionary that stores explainer classes,
            where the keys are strings representing the names or identifiers of the explainers, and the values
            are the respective classes (not instances) implementing the explainers.
        """
        # Store explainer classes (not instances) in a registry
        self.explainers: dict[str, Type[GenericExplainerAdapter]] = {}

    def register_explainer(self, name: str, explainer_cls: Type[GenericExplainerAdapter]):
        """
        Registers a new explainer class to the explainer registry.

        The provided class must inherit from `GenericExplainerAdapter`. Once registered,
        the explainer can be referenced by the given name for future use.

        Parameters:
        name: str
            The name under which the explainer will be registered.
        explainer_cls: Type[GenericExplainerAdapter]
            The class of the explainer to be registered. This class must be a subclass
            of `GenericExplainerAdapter`.
        Raises:
        TypeError
            If the provided `explainer_cls` does not inherit from `GenericExplainerAdapter`.
        """
        # Validate that the class inherits from GenericExplainerAdapter
        if not issubclass(explainer_cls, GenericExplainerAdapter):
            raise TypeError(
                f"Cannot register {explainer_cls.__name__}: must inherit from GenericExplainerAdapter"
            )
        # Register the explainer class in the registry
        self.explainers[name] = explainer_cls
        logger.info(f"Explainer '{name}' registered successfully")

    def get_explainer(self, name: str, model, dataset) -> GenericExplainerAdapter:
        """
        Retrieves and initializes an explainer adapter based on its registered name.

        This method checks if an explainer with the specified name is registered. If the
        explainer is found, it initializes and returns the corresponding adapter.
        Otherwise, it raises an error indicating that the explainer is not registered.

        Parameters:
        name: str
            The name of the explainer to retrieve.
        model
            The model to be explained by the explainer.
        dataset
            The dataset to be used for the explanation process.

        Returns:
        GenericExplainerAdapter
            An instance of the registered explainer adapter.

        Raises:
        KeyError
            If no explainer is registered under the specified name.
        """
        if name not in self.explainers:
            raise KeyError(f"Unregistered explainer '{name}'.")
        return self._create_explainer(name, model, dataset)

    def list_compatible_explainers(self, dataset_type: str, model_type: str) -> List[str]:
        """
        Filters and returns a list of compatible explainer names for a given dataset type and
        model type.

        The method checks the compatibility of each explainer defined in the 'explainers'
        dictionary by invoking their class-level method `is_compatible()`. The names of the
        compatible explainers are collected and returned as a list. A log entry is generated
        indicating the number of compatible explainers found for the provided combination
        of dataset and model types.

        Args:
            dataset_type: The type of dataset to check compatibility for.
            model_type: The type of model to check compatibility for.

        Returns:
            List of explainer names that are compatible with the specified dataset type and
            model type.
        """
        # Filter explainers using their is_compatible() class method
        compatible = [
            name for name, cls in self.explainers.items()
            if cls.is_compatible(dataset_type, model_type)
        ]
        logger.info(f"Found {len(compatible)} compatible explainer(s) for {dataset_type} + {model_type}")
        return compatible

    def explain_instance(self, explainer_name: str, model, data: Any, dataset_type: str, instance,
                         class_name: str = None):
        """
        Generates an explanation for a single instance using a specified explainer.

        This method allows the user to generate explanations for a specific data instance
        by selecting an explainer from the available registered explainers. It performs
        several steps including dataset creation, explainer compatibility checks, and
        finally invoking the explanation generation.

        Parameters:
            explainer_name (str): The name of the registered explainer to be used.
            model: The machine learning model for which the explanation is being generated.
            data (Any): The raw dataset or input data used for creating the explainer input.
            dataset_type (str): The type of the dataset (e.g., 'tabular', 'image', 'text').
            instance: The specific instance of the data to explain.
            class_name (str, optional): Name of a specific class or label to consider during
                                        explanation generation, if applicable.

        Raises:
            KeyError: If the specified explainer is not registered.
            ValueError: If the selected explainer is not compatible with the dataset type
                        and/or the model type.

        Returns:
            Explanation object generated by the explainer, detailing the explanation for
            the given data instance.
        """
        # Step 0: create the dataset from raw data
        dataset = self._create_dataset(data, dataset_type, class_name)

        # Step 1: check if the explainer exists
        if explainer_name not in self.explainers:
            raise KeyError(f"Explainer '{explainer_name}' not registered")

        # Step 2: instantiate explainer
        explainer_cls = self.explainers[explainer_name]
        explainer = explainer_cls(model=model, dataset=dataset)

        # Step 3: compatibility check
        if not explainer.is_compatible(dataset_type=dataset_type, model_type=type(model).__name__):
            raise ValueError(
                f"Explainer '{explainer_name}' is not compatible with dataset type '{dataset_type}' and model type '{type(model).__name__}'")

        # Step 4: generate explanation
        explanation = explainer.explain_instance(instance)
        return explanation

    def explain_global(self, explainer_name: str, model, dataset):
        """
        Executes a global explanation process using the specified explainer for a given model and dataset.

        GLOBAL EXPLANATION: Generates explanation for the entire model behavior
        WORKFLOW: Validate → Create → Check Compatibility → Explain → Visualize

        This method retrieves an explainer class by its name, initializes it with the provided model
        and dataset, and checks compatibility between the explainer, model, and dataset. If compatible,
        it generates the global explanation and optionally visualizes it if the explainer supports
        visualization.

        Parameters:
        explainer_name: str
            The name of the explainer to be used for explaining the model.
        model
            The machine learning model to be explained.
        dataset
            The dataset used for generating explanations.
        Raises:
        KeyError
            If the specified explainer name is not registered.
        ValueError
            If the explainer is not compatible with the given model and dataset types.
        Returns:
        The global explanation generated by the explainer.
        """
        # Step 1: Validate that explainer is registered
        self._validate_explainer_exists(explainer_name)

        # Step 2: Create explainer instance with model and dataset
        explainer = self._create_explainer(explainer_name, model, dataset)

        # Step 3: Validate compatibility between explainer, model, and dataset
        self._validate_compatibility(explainer_name, explainer, model, dataset)

        # Step 4: Generate the global explanation for the model
        logger.info(f"Generating global explanation using '{explainer_name}'")
        explanation = explainer.explain_global()

        # Step 5: Visualize if the explainer supports it
        self._visualize_if_supported(explainer, explanation)

        return explanation

    # ============================================================================
    # PRIVATE HELPER METHODS - Validation, Creation, and Utility Functions
    # ============================================================================

    def _validate_explainer_exists(self, explainer_name: str) -> None:
        """
        Validates that the explainer with the given name is registered.

        Args:
            explainer_name (str): The name of the explainer to validate.
        Raises:
            KeyError: If the explainer is not registered.
        """
        # Check if explainer name exists in the registry
        if explainer_name not in self.explainers:
            raise KeyError(f"Explainer '{explainer_name}' not registered")

    def _create_dataset(self, data: Any, dataset_type: str, class_name: str = None):
        """
        Creates and returns a dataset instance using the DatasetFactory class. The dataset
        is initialized based on the provided data, dataset type, and optionally, a class name.

        Parameters:
            data: Any
                The input data to be passed to the factory for creating the dataset.
            dataset_type: str
                The type of dataset to be created.
            class_name: str, optional
                The optional class name to specify the dataset structure.

        Returns:
            Dataset
                A dataset instance created by the DatasetFactory.

        Raises:
            This method may raise exceptions if the dataset creation process fails or if
            invalid parameters are provided.
        """
        dataset = DatasetFactory.create(data=data, dataset_type=dataset_type, class_name=class_name)
        return dataset

    def _create_explainer(self, name: str, model, dataset) -> GenericExplainerAdapter:
        """
        Creates and returns an instance of a specific explainer class associated with the given name.
        This method retrieves the explainer class from the `explainers` dictionary using the provided
        name and then initializes it with the given model and dataset.

        Arguments:
            name: str
                The name identifying the explainer class to create.
            model
                The model to be explained. Type is inferred from the context.
            dataset
                The dataset associated with the model. Type is inferred from the context.

        Returns:
            GenericExplainerAdapter
                An instance of the explainer class initialized with the given model and dataset.
        """
        explainer_cls = self.explainers[name]  # Type[GenericExplainerAdapter]
        return explainer_cls(model=model, dataset=dataset)  # returns an instance of the explainer

    def _validate_compatibility(self, explainer_name: str, explainer: GenericExplainerAdapter,
                                model, dataset) -> None:
        """
        Validates compatibility of the provided explainer with the given model and dataset.

        This method checks if the specified explainer adapter is compatible with the given
        dataset and model by comparing their type names. Compatibility is determined using
        the `is_compatible` method of the provided explainer.

        Args:
            explainer_name: Name of the explainer as a string.
            explainer: Object of type GenericExplainerAdapter, responsible for verifying compatibility.
            model: The machine learning model to be checked for compatibility.
            dataset: The dataset to be checked for compatibility.

        Raises:
            ValueError: If the explainer is not compatible with the given dataset and model.
        """
        # Extract type names and convert to lowercase for comparison
        dataset_type = type(dataset).__name__.lower()
        model_type = type(model).__name__.lower()

        # Check compatibility using the explainer's is_compatible method
        if not explainer.is_compatible(dataset_type, model_type):
            raise ValueError(
                f"Explainer '{explainer_name}' not compatible with {dataset_type} and {model_type}"
            )
        logger.info(f"Compatibility check passed: '{explainer_name}' ✓ {dataset_type} ✓ {model_type}")

    def _visualize_if_supported(self, explainer: GenericExplainerAdapter, explanation) -> None:
        """
        Visualizes the explanation if the given explainer supports visualization.

        The method checks if the provided explainer has a "visualize" method by using
        duck typing. If the method is available, it invokes the "visualize" method of
        the explainer with the provided explanation. Otherwise, a message indicating
        lack of visualization support is printed.

        Parameters:
            explainer: GenericExplainerAdapter
                The explainer object to be used for visualizing the explanation. It
                should have an attribute or method named "visualize" to support
                visualization.
            explanation
                The explanation data to be visualized. The specific type and structure
                of this data depend on the explainer being used.

        Returns:
            None
        """
        # Check if the explainer has a visualize method (duck typing)
        if hasattr(explainer, "visualize"):
            logger.info(f"Visualizing explanation with {explainer.explainer_name}")
            explainer.visualize(explanation)
        else:
            logger.error(f"Explainer does not support visualization")
