"""
Author: Xuyen Nguyen

This is a Decision Tree class that allows for building decision trees from training data and making predictions on new inputs.
"""

class DecisionTree:
  def __init__(self, maxDepth = None, minSplit=2, criterion='entropy'):
    """
    Initialize a decision tree.

    Parameters:
    - maxDepth (int or None): The maximum depth of the tree. If None, the tree grows until all leaves are pure or contains fewer than minSplit samples.
    - minSplit (int): The minimum number of samples required to split on an internal node.
    - criterion (str): The criterion used for spitting ('entropy','gini_index', or 'majority_error')
    """
    self.maxDepth = maxDepth
    self.criterion = criterion
    self.root = None

  def fit(self, X, y):
    """
    Build the decision tree using the provided training data.

    Parameters:
    - X (array-like, shape = [n_samples, n_features]): Training data.
    - y (array-like, shape = [n_samples]): Training lables.
    """
    pass

  def predict(self, X):
    """
    Predict the labels for the given data.

    Parameters:
    - X (array-like, shape = [n_samples, n_features]): Data for which to make predictions.

    Returns:
    - preidictions (array-like, shape = [n_shamples]): Predicted values.
    """
    pass

  def _build_tree(self, X, y, depth):
    """
      Recursively build the decision tree.

      Parameters:
      - X (array-like, shape = [n_samples, n_features]): Subset of training data.
      - y (array-like, shape = [n_samples]): Subset of training lables.
      - depth (int): Current depth in the tree.

      Returns:
      - node: The constructed decision tree node.

    """
    pass

  def _entropy(self, y, filter):
    """
    Calculate the entropy of a set of target values.

    Parameters:
    - y (array-like, shape = [n)samples]): labels.
    - filter (tuple of the form (attribute_name, attribute_value) or None): Specifies the specific value of the specific attribute whose entropy to be calculated. The type of attribute_name is str and that of the attribute_value is any. If None, calculate the overall entropy of the entire set.

    """
    pass

  def _gini_index(self, y, filter=None):
    """
    Calculate the gini index value of a set of target values.

    Parameters:
    - y (array-like, shape = [n)samples]): labels.
    - filter (tuple of the form (attribute_name, attribute_value) or None): Specifies the specific value of the specific attribute whose gini index value to be calculated. The type of attribute_name is str and that of the attribute_value is any. If None, calculate the overall gini index value of the entire set.

    """
    pass

  def _majority_error(self, y, filter=None):
    """
    Calculate the majority error of a set of target values.

    Parameters:
    - y (array-like, shape = [n)samples]): labels.
    - filter (tuple of the form (attribute_name, attribute_value) or None): Specifies the specific value of the specific attribute whose majority error to be calculated. The type of attribute_name is str and that of the attribute_value is any. If None, calculate the overall majority error of the entire set.

    """
    pass
