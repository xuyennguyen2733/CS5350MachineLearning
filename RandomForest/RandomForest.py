from DecisionTree import DecisionTree
import numpy as np

class RandomForest:
  def __init__(self, possible_values, max_depth=100, criterion = 'entropy', unknown=None, num_trees = 10, num_features=3):
    self.possible_values = possible_values
    self.criterion = criterion
    self.unknown = unknown
    self.num_trees = num_trees
    self.max_depth = max_depth
    self.num_features = num_features
    self.trees = []

  def train(self, X, y):
    self.trees = []
    for n in range(self.num_trees):
      pass
      # create bootstrap data
      X_bootstrap, y_bootstrap = self._create_boostrap_data(X, y)

      # build tree with boostrap data and number of selected features
      tree = DecisionTree(self.possible_values, self.max_depth, self.criterion, self.unknown)
      root = tree.ID3(X_bootstrap, y_bootstrap, True, self.num_features)

      # save tree
      self.trees.append(tree)

  def predict(self, X):
    votes = {}
    for tree in self.trees:
      prediction = tree.predict(X)
      votes[prediction] = votes.get(prediction,0) + 1
    return max(votes, key=votes.get)
    

  def _create_boostrap_data(self, X, y):
    n = len(y)
    random_indices = np.random.choice(n, size=n, replace=True)
    return X[random_indices], y[random_indices]