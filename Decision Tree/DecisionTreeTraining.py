import numpy as np
from DecisionTree import DecisionTree

# # # PREDICTION REPORT FOR CAR
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

# # PREDICTION REPORT FOR BANK

# column_headers = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome', 'label']
# possibleValues = {
# 'age': [0],
# 'job': ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"],
# 'marital': ["married","divorced","single"],
# 'education': ["unknown","secondary","primary","tertiary"],
# 'default': ["yes","no"],
# 'balance': [0],
# 'housing': ["yes","no"],
# 'loan': ["yes","no"],
# 'contact': ["unknown","telephone","cellular"],
# 'day': [0],
# 'month': ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
# 'duration': [0],
# 'campaign': [0],
# 'pdays': [0],
# 'previous': [0],
# 'poutcome': ["unknown","other","failure","success"]
# }

# data = np.genfromtxt(".\\bank-4\\train.csv", dtype=None, delimiter=",", names=column_headers, encoding=None)

# x_labels = []
# num_features = len(column_headers)-1
# for i in range(0,num_features):
#   x_labels.append( column_headers[i])

# y_label = column_headers[-1]

# X_train = data[x_labels]
# y_train = data[y_label]

# myTrees = []
# labels, count = np.unique(y_train, return_counts=True)
# accurate = 0

# data = np.genfromtxt(".\\bank-4\\test.csv", dtype=None, delimiter=",", names=column_headers, encoding=None)
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
# print('QUESTION 3 - WORKING WITH BANK DATA')
# print('(a)')
# print('                    Information_Gain     Majority_Error     Gini_Index')
# for impurity_method in ['entropy', 'gini_index', 'majority_error']:
#   accuracy_train = 0
#   accuracy_test = 0
#   for depth in range(1,17):
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
#   report['training_error'][impurity_method] = '{:.4f}%'.format(100-(100*accuracy_train/17))
#   report['testing_error'][impurity_method] = '{:.4f}%'.format(100-(100*accuracy_test/17))

# for row in report:
#   report_line = row + '        ' + report[row]['entropy'] + '             ' + report[row]['majority_error'] + '           ' + report[row]['gini_index']
#   print(report_line)

# print('(b)')
# print('                    Information_Gain     Majority_Error     Gini_Index')
# for impurity_method in ['entropy', 'gini_index', 'majority_error']:
#   accuracy_train = 0
#   accuracy_test = 0
#   for depth in range(1,17):
#     myTree = DecisionTree(possibleValues, depth, criterion=impurity_method, unknownIsMising=True)
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
#   report['training_error'][impurity_method] = '{:.4f}%'.format(100-(100*accuracy_train/17))
#   report['testing_error'][impurity_method] = '{:.4f}%'.format(100-(100*accuracy_test/17))

# for row in report:
#   report_line = row + '        ' + report[row]['entropy'] + '             ' + report[row]['majority_error'] + '           ' + report[row]['gini_index']
#   print(report_line)

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