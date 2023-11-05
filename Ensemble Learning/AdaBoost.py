"""
Author: Xuyen Nguyen

This is a Decision Tree class that allows for building decision trees from training data and making predictions on new inputs.
"""
import numpy as np
import sys
import math

class DecisionTree:
  def __init__(self, possibleValues, maxDepth = None, criterion='entropy', unknown=None):
    """
    Initialize a decision tree.

    Parameters:
    - possibleValues (dict<str,str or int>): All possible values of the possible attributes.
    - maxDepth (int or None): The maximum depth of the tree. If None, the tree grows until all leaves are pure or contains fewer than minSplit samples.
    - criterion (str): The criterion used for splitting ('entropy','gini_index', or 'majority_error')
    - unknown (string): The string that should be treated as missing value. If None, count them as their own value category.
    """

    self.possibleValues = possibleValues
    if maxDepth == None:
      self.maxDepth = sys.maxsize
    else:
      self.maxDepth = maxDepth
    if criterion in ['entropy', 'gini_index', 'majority_error']:   
      self.criterion = criterion 
    else:
      raise ValueError('{} is not a valid criterion!'.format(criterion))
    self.root = None
    self.threshold = {}
    self.unknown = unknown
    self.majorityLabel = {}


  def ID3(self, X, y):
    """
    Build the decision tree using the provided training data.

    Parameters:
    - X (array-like, shape = [n_samples, n_features]): Training data.
    - y (array-like, shape = [n_samples]): Training lables.
    - attributes (array-like, shape = [n_features]): the attributes being considered

    Returns:
    - root (Node): The root node of the decision tree
    """
    self.attributes = X.dtype.names

    if (self.unknown != None):
      for attribute in self.possibleValues:
        if isinstance(self.possibleValues[attribute][0], str):
          unique_labels, unique_count = np.unique(X[attribute], return_counts=True)
          max_label = unique_labels[np.argmax(unique_count)]
          for i in range(0,len(X[attribute])):
            X[attribute][i] = max_label
          self.majorityLabel[attribute] = max_label

    layer = 0

    if len(np.unique(y)) == 1:
      node = Node(y[0])
      return node
    
    if len(self.attributes) == 0 or layer == self.maxDepth:

      unique_labels, unique_count = np.unique(y, return_counts=True)
      max_label = unique_labels[np.argmax(unique_count)]
      node = Node(max_label)
      return node
    
    if (self.criterion == 'entropy'):
      self.impurityFunc = self._entropy
    elif (self.criterion == 'gini_index'):
      self.impurityFunc = self._gini_index
    else:
      self.impurityFunc = self._majority_error

    bestSplitAttribute = self._split(X, y, self.attributes)

    if (isinstance(self.possibleValues[bestSplitAttribute][0],(int,float))):
      bestSplitValues = np.unique(X[bestSplitAttribute])
      self.threshold[bestSplitAttribute] = np.mean(np.array(bestSplitValues))
      root = Node(bestSplitAttribute)

      # split less than or equal
      v = True
      root.addChild(v)
      X_v = X[X[bestSplitAttribute] <= self.threshold[bestSplitAttribute]]
      y_v = y[X[bestSplitAttribute] <= self.threshold[bestSplitAttribute]]
      if (len(X_v)==0):
        unique_labels, unique_count = np.unique(y, return_counts=True)
        max_label = unique_labels[np.argmax(unique_count)]
        node = Node(max_label)
        root.addChild(v,node)
      else:
        newAttributes = np.array(self.attributes)
        root.addChild(v, self._ID3_build(X_v, y_v, layer+1, newAttributes[newAttributes != bestSplitAttribute]))

      # split greater than
      v = False
      root.addChild(v)
      X_v = X[X[bestSplitAttribute] > self.threshold[bestSplitAttribute]]
      y_v = y[X[bestSplitAttribute] > self.threshold[bestSplitAttribute]]
      if (len(X_v)==0):
        unique_labels, unique_count = np.unique(y, return_counts=True)
        max_label = unique_labels[np.argmax(unique_count)]
        node = Node(max_label)
        root.addChild(v,node)
      else:
        newAttributes = np.array(self.attributes)
        root.addChild(v, self._ID3_build(X_v, y_v, layer+1, newAttributes[newAttributes != bestSplitAttribute]))
    else:
      bestSplitValues = np.unique(self.possibleValues[bestSplitAttribute])
      root = Node(bestSplitAttribute)
      for v in bestSplitValues:
        root.addChild(v)
        X_v = X[X[bestSplitAttribute]==v]
        y_v = y[X[bestSplitAttribute]==v]
        if (len(X_v)==0):
          unique_labels, unique_count = np.unique(y, return_counts=True)
          max_label = unique_labels[np.argmax(unique_count)]
          node = Node(max_label)
          root.addChild(v,node)
        else:
          newAttributes = np.array(self.attributes)
          root.addChild(v, self._ID3_build(X_v, y_v, layer+1, newAttributes[newAttributes != bestSplitAttribute]))
    self.root = root
    return root
  
  def _ID3_build(self, X, y, layer, attributes):
    """
    Build the decision tree using the provided training data.

    Parameters:
    - X (array-like, shape = [n_samples, n_features]): Training data.
    - y (array-like, shape = [n_samples]): Training lables.
    - layer (int): the current layer that the node is on.
    - attributes (array-like, shape = [n_feature]): the attributes being considered.

    Returns:
    - root (Node): root of the decision sub-tree
    """
    if len(np.unique(y)) == 1:
      node = Node(y[0])
      return node
    
    if len(attributes) == 0 or layer == self.maxDepth:
      unique_labels, unique_count = np.unique(y, return_counts=True)
      max_label = unique_labels[np.argmax(unique_count)]
      node = Node(max_label)
      return node

    bestSplitAttribute = self._split(X, y, attributes)

    if (isinstance(self.possibleValues[bestSplitAttribute][0],(int,float))):
      bestSplitValues = np.unique(X[bestSplitAttribute])
      self.threshold[bestSplitAttribute] = np.mean(np.array(bestSplitValues))
      root = Node(bestSplitAttribute)

      # split less than
      v = True
      root.addChild(v)
      X_v = X[X[bestSplitAttribute] <= self.threshold[bestSplitAttribute]]
      y_v = y[X[bestSplitAttribute] <= self.threshold[bestSplitAttribute]]
      if (len(X_v)==0):
        unique_labels, unique_count = np.unique(y, return_counts=True)
        max_label = unique_labels[np.argmax(unique_count)]
        node = Node(max_label)
        root.addChild(v,node)
      else:
        newAttributes = np.array(attributes)
        root.addChild(v, self._ID3_build(X_v, y_v, layer+1, newAttributes[newAttributes != bestSplitAttribute]))

      # split greater than
      v = False
      root.addChild(v)
      X_v = X[X[bestSplitAttribute] > self.threshold[bestSplitAttribute]]
      y_v = y[X[bestSplitAttribute] > self.threshold[bestSplitAttribute]]
      if (len(X_v)==0):
        unique_labels, unique_count = np.unique(y, return_counts=True)
        max_label = unique_labels[np.argmax(unique_count)]
        node = Node(max_label)
        root.addChild(v,node)
      else:
        newAttributes = np.array(attributes)
        root.addChild(v, self._ID3_build(X_v, y_v, layer+1, newAttributes[newAttributes != bestSplitAttribute]))
    else:
      bestSplitValues = np.unique(self.possibleValues[bestSplitAttribute])
      root = Node(bestSplitAttribute)
      for v in bestSplitValues:
        root.addChild(v)
        X_v = X[X[bestSplitAttribute]==v]
        y_v = y[X[bestSplitAttribute]==v]
        if (len(X_v)==0):
          unique_labels, unique_count = np.unique(y, return_counts=True)
          max_label = unique_labels[np.argmax(unique_count)]
          node = Node(max_label)
          root.addChild(v,node)
        else:
          newAttributes = np.array(attributes)
          root.addChild(v, self._ID3_build(X_v, y_v, layer+1, newAttributes[newAttributes != bestSplitAttribute]))
    return root

  def _split(self, X, y, attributes):
    '''
    Calculate gains and select the attribute with the most gain

    Parameters:
    - X (array-like, shape = [n_samples, n_features]): Input data.
    - y (array-like, shape = [n_samples]): Expected labels.
    - attributes (array-like, shape = [n_features]): all possible attributes.

    Returns:
    - attribute (str): the attribute with the most gain
    '''
    impurityVal = self.impurityFunc(y)
    attribGains = []
    total = len(y)
    for a in attributes:
      values, count = np.unique(X[a], return_counts=True)
      gain = impurityVal
      for v in values:
        val_purity = self.impurityFunc(y[X[a]==v])
        val_count = count[values == v]
        gain -= (val_count/total)*val_purity

      attribGains.append(gain)
    return attributes[np.argmax(attribGains)]

  def predict(self, X):
    """
    Predict the labels for the given data.

    Parameters:
    - X (array-like, shape = [n_samples, n_features]): Data for which to make predictions.

    Returns:
    - preidiction (str): Predicted value.
    """
    if self.unknown != None:
      for attribute in X.dtype.names:
        if (isinstance(self.possibleValues[attribute][0],str) and X[attribute]==self.unknown):
          X[attribute] = self.majorityLabel[attribute]
    # print(X)
    splitAttribute = self.root.attribute
    if isinstance(self.possibleValues[splitAttribute][0], (float, int)):
      splitLabel = X[splitAttribute] <= self.threshold[splitAttribute]
    else:
      splitLabel = X[splitAttribute]
    currentNode = self.root.children[splitLabel]
    while (currentNode.children != None):
      splitAttribute = currentNode.attribute
      if isinstance(self.possibleValues[splitAttribute][0], (float, int)):
        splitLabel = X[splitAttribute] <= self.threshold[splitAttribute]
      else:
        splitLabel = X[splitAttribute]
      currentNode = currentNode.children[splitLabel]
    return currentNode.attribute

  def _entropy(self, y):
    """
    Calculate the entropy of a set of target values.

    Parameters:
    - y (array-like, shape = [n)samples]): labels.
    
    """

    unique_value = np.unique(y)
    total = len(y)
    entropy = 0
    for val in unique_value:
      count = len(y[y == val])
      entropy -= (count/total)*math.log2(count/total)

    return entropy

  def _gini_index(self, y):
    """
    Calculate the gini index value of a set of target values.

    Parameters:
    - y (array-like, shape = [n)samples]): labels.


    """
    unique_value = np.unique(y)
    total = len(y)
    gini = 1
    for val in unique_value:
      count = len(y[y == val])
      gini -= pow((count/total),2)

    return gini

  def _majority_error(self, y):
    """
    Calculate the majority error of a set of target values.

    Parameters:
    - y (array-like, shape = [n)samples]): labels.
    
    """
    unique_value = np.unique(y)
    total = len(y)
    maxCount = 0
    for val in unique_value:
      count = len(y[y==val])
      maxCount = max(maxCount,count)
    return (total-maxCount)/total
  
  def AdaBoost(self, X, y, T):
    threshold = 1e-10
    # adaBoostTree = DecisionTree(self.possibleValues,1)
    num_examples = 10
    weights = np.full((T+1,num_examples), 1/num_examples, dtype=float)
    # weights = np.full((T+1,len(y)), 1/len(y), dtype=float)
    errors = np.zeros(T)
    votes = np.zeros(T)
    votes = np.zeros(T)
    self.stumps = np.full(T, None)
    for t in range(0,T):
      self.stumps[t] = DecisionTree(self.possibleValues,1)
      
      self.stumps[t].ID3(X,y)
      y_predict = np.zeros(len(y))
      for i in range (0, len(y)):
        y_predict[i] = self.stumps[t].predict(X[i])
        if y[i] != y_predict[i]:
          errors[t] += weights[t][i]
      y_predict = np.zeros(num_examples)
      if errors[t] == 0:
        errors[t] = threshold
      votes[t] = 1/2 * math.log((1-errors[t])/errors[t])
      for i in range(0, num_examples):
        weights[t+1][i] = weights[t][i] * math.exp(-votes[t] * y[i] * y_predict[i])
      for i in range(0, num_examples):
        weights[t+1][i] = weights[t][i] * math.exp(-votes[t] * subY[i] * y_predict[i])
      weights[t+1] = weights[t+1]/np.sum(weights[t+1])
    self.stumps_dict[T] = self.stumps
    self.votes = votes
    self.errors = errors
    self.weights = weights 
  
  def AdaBoostPredict(self, X):
    final = 0
    for stump, vote in zip(self.stumps, self.votes):
      final += vote * stump.predict(X)
    return np.sign(final)
  
  def BaggedTree(self, X, y):
      random_indices = np.random.choice(len(X), size=num_examples, replace=False)
      subX = X[random_indices]
      subY = y[random_indices]

class Node:
  def __init__(self, attribute=''):
    self.attribute = attribute
    self.children = None

  def addChild(self, edgeLabel, childNode=None):
    if self.children == None:
      self.children = {}
    self.children[edgeLabel] = childNode

  def next(self, childAttribute):
    return self.children[childAttribute]