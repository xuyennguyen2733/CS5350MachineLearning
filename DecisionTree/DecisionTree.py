"""
Author: Xuyen Nguyen

This is a Decision Tree class that allows for building decision trees from training data and making predictions on new inputs.
"""

class DecisionTree:
  def __init__(self, possibleValues, maxDepth = None, criterion='entropy', unknownIsMising=False):
    """
    Initialize a decision tree.

    Parameters:
    - possibleValues (dict<str,str or int>): All possible values of the possible attributes.
    - maxDepth (int or None): The maximum depth of the tree. If None, the tree grows until all leaves are pure or contains fewer than minSplit samples.
    - criterion (str): The criterion used for spitting ('entropy','gini_index', or 'majority_error')
    - unknownIsMissing (bool): Whether to treat values of 'unknown' as missing data or not. If False, count them as their own value category.
    """
    import sys
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
    self.unknownIsMissing = unknownIsMising
    self.majorityLabel = {}


  def ID3(self, X, y, attributes=None):
    """
    Build the decision tree using the provided training data.

    Parameters:
    - X (array-like, shape = [n_samples, n_features]): Training data.
    - y (array-like, shape = [n_samples]): Training lables.
    - attributes (array-like, shape = [n_features]): the attributes being considered

    Returns:
    - root (Node): The root node of the decision tree
    """
    import numpy as np

    if (self.unknownIsMissing):
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
    
    if len(attributes) == 0 or layer == self.maxDepth:

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

    bestSplitAttribute = self._split(X, y, attributes)

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
    self.root = root
    return root
  
  def _ID3_build(self, X, y, layer, attributes=None):
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
    import numpy as np

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
    if self.unknownIsMissing:
      for attribute in X.dtype.names:
        if (isinstance(self.possibleValues[attribute][0],str) and X[attribute]=='unknown'):
          X[attribute] = self.majorityLabel[attribute]

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
    import numpy as np
    import math
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
    import numpy as np
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
    import numpy as np
    unique_value = np.unique(y)
    total = len(y)
    maxCount = 0
    for val in unique_value:
      count = len(y[y==val])
      maxCount = max(maxCount,count)
    return (total-maxCount)/total
    

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

# # PREDICTION REPORT FOR CAR
column_headers = ['buying','maint','doors','persons','lug_boot','safety','label']
possibleValues = {
  'buying':   ['vhigh', 'high', 'med', 'low'],
  'maint':    ['vhigh', 'high', 'med', 'low'],
  'doors':    ['2', '3', '4', '5more'],
  'persons':  ['2', '4', 'more'],
  'lug_boot': ['small', 'med', 'big'],
  'safety':   ['low', 'med', 'high'],
}

data = np.genfromtxt(".\\car-4\\train.csv", dtype=None, delimiter=",", names=column_headers, encoding=None)

x_labels = []
num_features = len(column_headers)-1
for i in range(0,num_features):
  x_labels.append( column_headers[i])

y_label = column_headers[-1]

X_train = data[x_labels]
y_train = data[y_label]

myTrees = []
labels, count = np.unique(y_train, return_counts=True)
accurate = 0

data = np.genfromtxt(".\\car-4\\test.csv", dtype=None, delimiter=",", names=column_headers, encoding=None)
X_test = data[x_labels]
y_test = data[y_label]
report = {
  'training_error': {
    'entropy': '',
    'gini_index': '',
    'majority_error': ''
  },
  'testing_error': {
    'entropy': '',
    'gini_index': '',
    'majority_error': ''
  }
}
print('QUESTION 2 - WORKING WITH CARS DATA')
print('                    Information_Gain     Majority_Error     Gini_Index')
for impurity_method in ['entropy', 'gini_index', 'majority_error']:
  accuracy_train = 0
  accuracy_test = 0
  for depth in range(1,7):
    myTree = DecisionTree(possibleValues, depth, criterion=impurity_method)
    root = myTree.ID3(X_train, y_train, x_labels)
    accurate_train = 0
    accurate_test = 0

    for i in range(0,len(y_train)):
      if (y_train[i]==myTree.predict(X_train[i])):
        accurate_train+=1

    accuracy_train += accurate_train/len(y_train)

    for i in range(0,len(y_test)):
      if (y_test[i]==myTree.predict(X_test[i])):
        accurate_test+=1

    accuracy_test += accurate_test/len(y_test)
    
  report['training_error'][impurity_method] = '{:.4f}%'.format(100-(100*accuracy_train/6))
  report['testing_error'][impurity_method] = '{:.4f}%'.format(100-(100*accuracy_test/6))

for row in report:
  report_line = row + '        ' + report[row]['entropy'] + '             ' + report[row]['majority_error'] + '           ' + report[row]['gini_index']
  print(report_line)

# PREDICTION REPORT FOR BANK

column_headers = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome', 'label']
possibleValues = {
'age': [0],
'job': ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"],
'marital': ["married","divorced","single"],
'education': ["unknown","secondary","primary","tertiary"],
'default': ["yes","no"],
'balance': [0],
'housing': ["yes","no"],
'loan': ["yes","no"],
'contact': ["unknown","telephone","cellular"],
'day': [0],
'month': ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
'duration': [0],
'campaign': [0],
'pdays': [0],
'previous': [0],
'poutcome': ["unknown","other","failure","success"]
}

data = np.genfromtxt(".\\bank-4\\train.csv", dtype=None, delimiter=",", names=column_headers, encoding=None)

x_labels = []
num_features = len(column_headers)-1
for i in range(0,num_features):
  x_labels.append( column_headers[i])

y_label = column_headers[-1]

X_train = data[x_labels]
y_train = data[y_label]

myTrees = []
labels, count = np.unique(y_train, return_counts=True)
accurate = 0

data = np.genfromtxt(".\\bank-4\\test.csv", dtype=None, delimiter=",", names=column_headers, encoding=None)
X_test = data[x_labels]
y_test = data[y_label]
report = {
  'training_error': {
    'entropy': '',
    'gini_index': '',
    'majority_error': ''
  },
  'testing_error': {
    'entropy': '',
    'gini_index': '',
    'majority_error': ''
  }
}
print('QUESTION 3 - WORKING WITH BANK DATA')
print('(a)')
print('                    Information_Gain     Majority_Error     Gini_Index')
for impurity_method in ['entropy', 'gini_index', 'majority_error']:
  accuracy_train = 0
  accuracy_test = 0
  for depth in range(1,17):
    myTree = DecisionTree(possibleValues, depth, criterion=impurity_method)
    root = myTree.ID3(X_train, y_train, x_labels)
    accurate_train = 0
    accurate_test = 0

    for i in range(0,len(y_train)):
      if (y_train[i]==myTree.predict(X_train[i])):
        accurate_train+=1

    accuracy_train += accurate_train/len(y_train)

    for i in range(0,len(y_test)):
      if (y_test[i]==myTree.predict(X_test[i])):
        accurate_test+=1

    accuracy_test += accurate_test/len(y_test)
  report['training_error'][impurity_method] = '{:.4f}%'.format(100-(100*accuracy_train/17))
  report['testing_error'][impurity_method] = '{:.4f}%'.format(100-(100*accuracy_test/17))

for row in report:
  report_line = row + '        ' + report[row]['entropy'] + '             ' + report[row]['majority_error'] + '           ' + report[row]['gini_index']
  print(report_line)

print('(b)')
print('                    Information_Gain     Majority_Error     Gini_Index')
for impurity_method in ['entropy', 'gini_index', 'majority_error']:
  accuracy_train = 0
  accuracy_test = 0
  for depth in range(1,17):
    myTree = DecisionTree(possibleValues, depth, criterion=impurity_method, unknownIsMising=True)
    root = myTree.ID3(X_train, y_train, x_labels)
    accurate_train = 0
    accurate_test = 0

    for i in range(0,len(y_train)):
      if (y_train[i]==myTree.predict(X_train[i])):
        accurate_train+=1

    accuracy_train += accurate_train/len(y_train)

    for i in range(0,len(y_test)):
      if (y_test[i]==myTree.predict(X_test[i])):
        accurate_test+=1

    accuracy_test += accurate_test/len(y_test)
  report['training_error'][impurity_method] = '{:.4f}%'.format(100-(100*accuracy_train/17))
  report['testing_error'][impurity_method] = '{:.4f}%'.format(100-(100*accuracy_test/17))

for row in report:
  report_line = row + '        ' + report[row]['entropy'] + '             ' + report[row]['majority_error'] + '           ' + report[row]['gini_index']
  print(report_line)
