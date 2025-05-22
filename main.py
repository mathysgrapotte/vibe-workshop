import datasets
import numpy as np
import logging
from functools import partial
from typing import Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_random_data(num_samples: int, num_features: int) -> datasets.Dataset:
    """This function generates a random dataset with a given number of samples and features.

    Args:
        num_samples (int): The number of samples in the dataset.
        num_features (int): The number of features in the dataset.

    Returns:
        datasets.Dataset: A random dataset with the given number of samples and features.
    """
    # Generate random data
    
    # Create random features
    features = {
        'feature_' + str(i): np.random.randn(num_samples) 
        for i in range(num_features)
    }
    
    # Create random labels (binary classification)
    labels = np.random.randint(0, 2, size=num_samples)
    
    # Create dataset
    dataset = datasets.Dataset.from_dict({
        **features,
        'label': labels
    })
    
    return dataset

def multiply_by_two(x: np.ndarray) -> np.ndarray:
    """This function multiplies an input numpy array by 2.
    
    Args:
        x (np.ndarray): The input numpy array.

    Returns:
        np.ndarray: The input numpy array multiplied by 2.
    """
    return x * 2

def map_function(batch: datasets.formatting.formatting.LazyBatch, feature_name: str, apply_function: Callable) -> dict:
    """This function applies a given function to a given feature in a batch.
    
    Args:
        batch (datasets.formatting.fromatting.LazyBatch): The batch to apply the function to.
        feature_name (str): The name of the feature to apply the function to.
        apply_function (Callable): The function to apply to the feature.
    
    Returns:
        dict: The batch with the function applied to the feature.
    """
    # Apply the function to each element in the batch
    batch[feature_name] = [apply_function(x) for x in batch[feature_name]]
    return batch


def main():
    dataset = generate_random_data(20, 10)
    logger.info(f"original dataset: {dataset}")
    logger.info(f"feature_0: {dataset['feature_0']}")
    function_to_map = partial(map_function, feature_name='feature_0', apply_function=multiply_by_two)
    dataset = dataset.map(function_to_map, batched=True, batch_size=2)
    logger.info(f"dataset after mapping: {dataset}")
    logger.info(f"feature_0: {dataset['feature_0']}")
if __name__ == "__main__":
    main()
