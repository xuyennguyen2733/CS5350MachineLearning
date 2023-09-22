"""
Author: Xuyen Nguyen

This is a Decision Tree class that allows for building decision trees from training data and making predictions on new inputs.
"""

class DecisionTree:
  def __init__(self, possibleValues, maxDepth = None, minSplit=2, criterion='entropy'):
    """
    Initialize a decision tree.

    Parameters:
    - maxDepth (int or None): The maximum depth of the tree. If None, the tree grows until all leaves are pure or contains fewer than minSplit samples.
    - minSplit (int): The minimum number of samples required to split on an internal node.
    - criterion (str): The criterion used for spitting ('entropy','gini_index', or 'majority_error')
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

  def ID3(self, X, y, attributes=None):
    """
    Build the decision tree using the provided training data.

    Parameters:
    - X (array-like, shape = [n_samples, n_features]): Training data.
    - y (array-like, shape = [n_samples]): Training lables.
    """
    import numpy as np

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

  def _gini_index(self, y):
    """
    Calculate the gini index value of a set of target values.

    Parameters:
    - y (array-like, shape = [n)samples]): labels.
    - filter (tuple of the form (attribute_name, attribute_value) or None): Specifies the specific value of the specific attribute whose gini index value to be calculated. The type of attribute_name is str and that of the attribute_value is any. If None, calculate the overall gini index value of the entire set.

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
    - filter (tuple of the form (attribute_name, attribute_value) or None): Specifies the specific value of the specific attribute whose majority error to be calculated. The type of attribute_name is str and that of the attribute_value is any. If None, calculate the overall majority error of the entire set.

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

# X = [
#   ['S','H','H','W'],
#   ['S','H','H','S'],
#   ['O','H','H','W'],
#   ['R','M','H','W'],
#   ['R','C','N','W'],
#   ['R','C','N','S'],
#   ['O','C','N','S'],
#   ['S','M','H','W'],
#   ['S','C','N','W'],
#   ['R','M','N','W'],
#   ['S','M','N','S'],
#   ['O','M','H','S'],
#   ['O','H','N','W'],
#   ['R','M','H','S']
# ]

# possibleValues = {
#   'Outlook': ['S','O','R'],
#   'Temperature': ['T','M','C'],
#   'Humidity': ['H','N','L'],
#   'Wind': ['S','W']
# }

# X = np.array(X)
# X = X.T
# attributes = ['Outlook','Temperature','Humidity','Wind']
# X = np.rec.fromarrays(X, names=attributes)

# # X.dtype.names = attributes
# y = ['-','-','+','+','+','-','+','-','+','+','+','+','+','-']
# y = np.array(y)

# myTree = DecisionTree()
# myTree.ID3(X,y,attributes,possibleValues)
# X[-1][2]='L'
# X[-1][0]='S'
# print(myTree.predict(X[-1]))

# PREDICTION REPORT FOR CAR
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
    'entropy': [],
    'gini_index': [],
    'majority_error': []
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

# '''
# 1 - age (numeric)
#    2 - job : type of job (categorical: ) 
#    3 - marital : marital status (categorical: ; note: "divorced" means divorced or widowed)
#    4 - education (categorical: )
#    5 - default: has credit in default? (binary: "yes","no")
#    6 - balance: average yearly balance, in euros (numeric) 
#    7 - housing: has housing loan? (binary: "yes","no")
#    8 - loan: has personal loan? (binary: "yes","no")
#    # related with the last contact of the current campaign:
#    9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
#   10 - day: last contact day of the month (numeric)
#   11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
#   12 - duration: last contact duration, in seconds (numeric)
#    # other attributes:
#   13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
#   14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
#   15 - previous: number of contacts performed before this campaign and for this client (numeric)
#   16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")
# '''

# column_headers = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']
# possibleValues = {
# 'age': [],
# 'job': ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"],
# 'marital': ["married","divorced","single"],
# 'education': [],
# 'default': [],
# 'balance': [],
# 'housing': [],
# 'loan': [],
# 'contact': [],
# 'day': [],
# 'month': [],
# 'duration': [],
# 'campaign': [],
# 'pdays': [],
# 'previous': [],
# 'poutcome': []
# }

# data = np.genfromtxt(".\\car-4\\train.csv", dtype=None, delimiter=",", names=column_headers, encoding=None)

# x_labels = []
# num_features = len(column_headers)-1
# for i in range(0,num_features):
#   x_labels.append( column_headers[i])

# y_label = column_headers[-1]

# X_train = data[x_labels]
# y_train = data[y_label]

# myTrees = []
# labels, count = np.unique(y_train, return_counts=True)
# # node = myTree.ID3(X_train, y_train, x_labels)
# accurate = 0

# data = np.genfromtxt(".\\car-4\\test.csv", dtype=None, delimiter=",", names=column_headers, encoding=None)
# X_test = data[x_labels]
# y_test = data[y_label]
# report = {
#   'training_error': {
#     'entropy': '',
#     'gini_index': '',
#     'majority_error': ''
#   },
#   'testing_error': {
#     'entropy': [],
#     'gini_index': [],
#     'majority_error': []
#   }
# }
# # print("y test",y_test)
# # print("x test", X_test[15])
# print('               Entropy     Gini_Index     Majority_error')
# for impurity_method in ['entropy', 'gini_index', 'majority_error']:
#   accuracy_train = 0
#   accuracy_test = 0
#   for depth in range(1,7):
#     # print('Decision tree with depth =',depth,'using', impurity_method)
#     myTree = DecisionTree(possibleValues, depth, criterion=impurity_method)
#     root = myTree.ID3(X_train, y_train, x_labels)
#     accurate_train = 0
#     accurate_test = 0

#     for i in range(0,len(y_train)):
#     # print('test number',i)
#       if (y_train[i]==myTree.predict(X_train[i])):
#         accurate_train+=1
#     # print('     Training Data Accuracy: {:.3f}%'.format(100*accurate_train/len(y_train)).replace('.000',''))

#     accuracy_train += accurate_train/len(y_train)

#     for i in range(0,len(y_test)):
#     # print('test number',i)
#       if (y_test[i]==myTree.predict(X_test[i])):
#         accurate_test+=1

#     accuracy_test += accurate_test/len(y_test)
    
#     # report['testing_error'][impurity_method].append(accurate_test/len(y_test))
#     # print('     Testing Data Accuracy: {:.3f}%'.format(100*accurate_test/len(y_test)).replace('.000',''))
#   report['training_error'][impurity_method] = '{:.4f}%'.format(100-(100*accuracy_train/6))
#   report['testing_error'][impurity_method] = '{:.4f}%'.format(100-(100*accuracy_test/6))


# # print(report)
# for row in report:
#   report_line = row + '   ' + report[row]['entropy'] + '       ' + report[row]['gini_index'] + '            ' + report[row]['majority_error']
#   pr

