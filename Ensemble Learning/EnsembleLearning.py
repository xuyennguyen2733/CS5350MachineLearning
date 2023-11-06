import numpy as np
from AdaBoost import DecisionTree

# # PREDICTION REPORT FOR CAR
# column_headers = ['buying','maint','doors','persons','lug_boot','safety','label']
# possibleValues = {
#   'buying':   ['vhigh', 'high', 'med', 'low'],
#   'maint':    ['vhigh', 'high', 'med', 'low'],
#   'doors':    ['2', '3', '4', '5more'],
#   'persons':  ['2', '4', 'more'],
#   'lug_boot': ['small', 'med', 'big'],
#   'safety':   ['low', 'med', 'high'],
# }

# data = np.genfromtxt(".\\car-4\\train.csv", dtype=None, delimiter=",", names=column_headers, encoding=None)

# x_labels = []
# num_features = len(column_headers)-1
# for i in range(0,num_features):
#   x_labels.append( column_headers[i])

# y_label = column_headers[-1]

# X_train = data[x_labels]
# y_train = data[y_label]

# print(X_train.dtype.names)

# myTrees = []
# labels, count = np.unique(y_train, return_counts=True)
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
#     'entropy': '',
#     'gini_index': '',
#     'majority_error': ''
#   }
# }
# print('QUESTION 2 - WORKING WITH CARS DATA')
# print('                    Information_Gain     Majority_Error     Gini_Index')
# for impurity_method in ['entropy', 'gini_index', 'majority_error']:
#   accuracy_train = 0
#   accuracy_test = 0
#   for depth in range(1,7):
#     myTree = DecisionTree(possibleValues, depth, criterion=impurity_method)
#     root = myTree.ID3(X_train, y_train)
#     accurate_train = 0
#     accurate_test = 0

#     for i in range(0,len(y_train)):
#       if (y_train[i]==myTree.predict(X_train[i])):
#         accurate_train+=1

#     accuracy_train += accurate_train/len(y_train)

#     for i in range(0,len(y_test)):
#       if (y_test[i]==myTree.predict(X_test[i])):
#         accurate_test+=1

#     accuracy_test += accurate_test/len(y_test)
    
#   report['training_error'][impurity_method] = '{:.4f}%'.format(100-(100*accuracy_train/6))
#   report['testing_error'][impurity_method] = '{:.4f}%'.format(100-(100*accuracy_test/6))

# for row in report:
#   report_line = row + '        ' + report[row]['entropy'] + '             ' + report[row]['majority_error'] + '           ' + report[row]['gini_index']
#   print(report_line)

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

data = np.genfromtxt("bank-4\\train.csv", dtype=None, delimiter=",", names=column_headers, encoding=None)

x_labels = []
num_features = len(column_headers)-1
for i in range(0,num_features):
  x_labels.append( column_headers[i])

y_label = column_headers[-1]

X_train = data[x_labels]
y_train_categorical = data[y_label]
map = {'no': -1, 'yes': 1}
y_train = np.vectorize(map.get)(y_train_categorical)

myTrees = []
labels, count = np.unique(y_train, return_counts=True)
accurate = 0

data = np.genfromtxt(".\\bank-4\\test.csv", dtype=None, delimiter=",", names=column_headers, encoding=None)
X_test = data[x_labels]
y_test_categorical = data[y_label]
y_test = np.vectorize(map.get)(y_test_categorical)
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

myTree = DecisionTree(possibleValues)

for T in range(1, 500):
  myTree.AdaBoost(X_train, y_train, T)
  accurracy = 0
  # print("T =",T)
  num_tests = len(y_test)
  for x, y in zip(X_test, y_test):
    y_hat = myTree.AdaBoostPredict(x)
    # print("     Prediction", y_hat, "true value", y)
    if (y_hat == y):
      accurracy += 1/num_tests
  print("T =",T," Acurracy =", accurracy)

# for T, stumps in myTree.stumps_dict.items():
#   train_len = len(y_train)
#   test_len = len(y_test)
#   count = 0
#   for stump in stumps:
#     train_error = 0
#     test_error = 0
#     for x, y in zip(X_train, y_train):
#       if (stump.predict(x) != y):
#         train_error += 1/train_len
#     for x, y in zip(X_test, y_test):
#       if (stump.predict(x) != y):
#         train_error += 1/test_len
#     count += 1
#     print("T =",T," stump",count,"training error:",train_error,"testing error",test_error)
