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

  def ID3(self, X, y, attributes=None):
    """
    Build the decision tree using the provided training data.

    Parameters:
    - X (array-like, shape = [n_samples, n_features]): Training data.
    - y (array-like, shape = [n_samples]): Training lables.
    """
    import numpy as np

    if len(np.unique(y)) == 1:
      node = Node(y[0])
      return node
    
    if len(attributes) == 0:
      unique_labels, unique_count = np.unique(y, return_counts=True)
      max_label = unique_labels[np.argmax(unique_count)]
      node = Node(max_label)
      return node
    
    bestSplitAttribute = self._split(X, y, attributes)
    bestSplitValues = np.unique(X[bestSplitAttribute])
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
        root.addChild(v, self._ID3_build(X_v, y_v, newAttributes[newAttributes != bestSplitAttribute]))
    self.root = root
    return root
  
  def _ID3_build(self, X, y, attributes=None):
    """
    Build the decision tree using the provided training data.

    Parameters:
    - X (array-like, shape = [n_samples, n_features]): Training data.
    - y (array-like, shape = [n_samples]): Training lables.
    """
    import numpy as np

    if len(np.unique(y)) == 1:
      node = Node(y[0])
      return node
    
    if len(attributes) == 0:
      unique_labels, unique_count = np.unique(y, return_counts=True)
      max_label = unique_labels[np.argmax(unique_count)]
      node = Node(max_label)
      return node
    
    bestSplitAttribute = self._split(X, y, attributes)
    bestSplitValues = np.unique(X[bestSplitAttribute])
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
        root.addChild(v, self._ID3_build(X_v, y_v, newAttributes[newAttributes != bestSplitAttribute]))

    return root

  def _split(self, X, y, attributes):
    currentEntropy = self._entropy(y)
    attribGains = []
    total = len(y)
    for a in attributes:
      values, count = np.unique(X[a], return_counts=True)
      gain = currentEntropy
      for v in values:
        val_entropy = self._entropy(y[X[a]==v])
        val_count = count[values == v]
        gain -= (val_count/total)*val_entropy

      attribGains.append(gain)
    return attributes[np.argmax(attribGains)]

  def predict(self, X):
    """
    Predict the labels for the given data.

    Parameters:
    - X (array-like, shape = [n_samples, n_features]): Data for which to make predictions.

    Returns:
    - preidictions (array-like, shape = [n_shamples]): Predicted values.
    """
    splitAttribute = self.root.attribute
    splitLabel = X[splitAttribute]
    currentNode = self.root.children[splitLabel]
    while (currentNode.children != None):
      splitAt = currentNode.attribute
      splitLabel = X[splitAt]
      currentNode = currentNode.children[splitLabel]

    return currentNode.attribute


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

  def _entropy(self, y):
    """
    Calculate the entropy of a set of target values.

    Parameters:
    - y (array-like, shape = [n)samples]): labels.
    - filter (tuple of the form (attribute_name, attribute_value) or None): Specifies the specific value of the specific attribute whose entropy to be calculated. The type of attribute_name is str and that of the attribute_value is any. If None, calculate the overall entropy of the entire set.

    """
    import numpy as np
    import math
    unique_value = np.unique(y)
    total = len(y)
    entropy = 0
    for val in unique_value:
      count = len(y[y == val])
      entropy -= (count/total)*math.log2(count/total)

    return entropy

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

import numpy as np

X = [
  ['S','H','H','W'],
  ['S','H','H','S'],
  ['O','H','H','W'],
  ['R','M','H','W'],
  ['R','C','N','W'],
  ['R','C','N','S'],
  ['O','C','N','S'],
  ['S','M','H','W'],
  ['S','C','N','W'],
  ['R','M','N','W'],
  ['S','M','N','S'],
  ['O','M','H','S'],
  ['O','H','N','W'],
  ['R','M','H','S']
]

X = np.array(X)
X = X.T
attributes = ['Outlook','Temperature','Humidity','Wind']
X = np.rec.fromarrays(X, names=attributes)

# X.dtype.names = attributes
y = ['-','-','+','+','+','-','+','-','+','+','+','+','+','-']
y = np.array(y)

myTree = DecisionTree()
root = myTree.ID3(X, y, attributes)
print(X[3])
print(myTree.predict(X[3]))


# column_headers = ['buying','maint','doors','persons','lug_boot','safety','label']
# data = np.genfromtxt(".\\car-4\\train.csv", dtype=None, delimiter=",", names=column_headers, encoding=None)

# x_labels = []
# num_features = len(column_headers)-1
# for i in range(0,num_features):
#   x_labels.append( column_headers[i])

# y_label = column_headers[-1]

# X = data[x_labels]
# y = data[y_label]

# myTree = DecisionTree()
# labels, count = np.unique(y, return_counts=True)
# node = myTree.ID3(X, y, x_labels)
# print()
# print(data[(data['buying']=='high') & (data['label']=='unacc')][0][:])


# import math
# total = 4
# plus = 3
# minus = total - plus
# entropy = -(plus/total)*math.log2(plus/total)-(minus/total)*math.log2(minus/total)

# def entropy(total, plus):
#   if total == plus or plus == 0:
#     return 0
#   minus = total - plus
#   entropy = -(plus/total)*math.log2(plus/total)-(minus/total)*math.log2(minus/total)

#   return entropy 

# data = {
#   'total': 5+5/14,
#   0: 2,
#   1: 3+5/14
# }
# data['entropy'] = entropy(data['total'], data[1])

# attributes = [('Temperature',[('H', 0, 0),('M', 3+5/14, 2+4/14),('C', 2, 1)]), ('Humidity',[('H', 2, 1), ('N', 3+5/14, 2+5/14),('L',0,0)]), ('Wind',[('W', 3+5/14, 3+5/14),('S', 2, 0)])]

# '''
# O   T   H   W   Play?
# R   M   H   W   +
# R   C   N   W   +
# R   C   N   S   -
# R   M   N   W   +
# R   M   H   S   -
# R   M   N   W   +       5/14
# '''

# maxGain = 0
# maxGainName = ''
# for attribute in attributes:
#   attributeName, values = attribute
#   data[attributeName] = {}
#   print('\\\\\\\\For', attributeName)
#   gainStr = '\\\\Information Gain: ${:.3f} '.format(data['entropy'])
#   gain = data['entropy']
  
#   for value in values:
#     valueName, total, plus = value
#     minus = total - plus
#     valueDict = {'total': total, 'entropy': entropy(total, plus), 0: minus, 1: plus}
#     data[attributeName][valueName] = valueDict
#     gainStr += '- (\\frac{{{:.3f}}}{{{:.3f}}})({:.3f}) '.format(total,data['total'],valueDict['entropy'])
#     gain -= valueDict['entropy']*(total/data['total'])
#     print('\\\\$H({}) = -(\\frac{{{:.3f}}}{{{:.3f}}})log_2(\\frac{{{:.3f}}}{{{:.3f}}}) - (\\frac{{{:.3f}}}{{{:.3f}}})log_2(\\frac{{{:.3f}}}{{{:.3f}}}) = {:.3f}$'.format(valueName, plus,total,plus,total,minus,total,minus,total,valueDict['entropy']).replace('.000',''))
  
#   if gain > maxGain:
#     maxGain = gain
#     maxGainName = attributeName

#   gainStr += '= {:.3}$'.format(gain) 
#   print(gainStr.replace('.000',''))
# print('\\\\The best feature is {}'.format(maxGainName))

