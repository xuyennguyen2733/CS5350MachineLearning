from DecisionTree import DecisionTree
import numpy as np

# # PREDICTION REPORT FOR INCOME
column_headers = ['age',	'workclass',	'fnlwgt',	'education',	'education_num',	'marital_status',	'occupation',	'relationship',	'race',	'sex',	'capital_gain',	'capital_loss',	'hours_per_week',	'native_country',	'MoreThan50k']
possibleValues = {}

data = np.genfromtxt(".\\income\\train.csv", dtype=None, delimiter=",", names=column_headers, skip_header=1, encoding=None)

for h in column_headers:
  values = data[h]

  if np.issubdtype(values.dtype, np.number):
    possibleValues[h] = [0]
  else:
    unique_values = np.unique(values)
    possibleValues[h] = unique_values.tolist()
# print(possibleValues)
x_labels = []
num_features = len(column_headers)-1
for i in range(0,num_features):
  x_labels.append( column_headers[i])
y_label = column_headers[-1]

X_train = data[x_labels]
y_train = data[y_label]

myTrees = []
# labels, count = np.unique(y_train, return_counts=True)
# accurate = 0

column_headers = ['ID', 'age',	'workclass',	'fnlwgt',	'education',	'education_num',	'marital_status',	'occupation',	'relationship',	'race',	'sex',	'capital_gain',	'capital_loss',	'hours_per_week',	'native_country']
# possibleValues = {}

data = np.genfromtxt(".\\income\\test.csv", dtype=None, delimiter=",", names=column_headers, skip_header=1, encoding=None)

X_test = data[x_labels]
# y_test = data[y_label]
IDs = data['ID']

incomeTree = DecisionTree(possibleValues, criterion='majority_error', unknown= '?')
incomeTree.ID3(X_train,y_train)
predictions = {
  'ID': [],
  'Prediction': []
}
for i in range(0,len(IDs)):
   X_test[i]['native_country']
   predictions['ID'].append(IDs[i])
   predictions['Prediction'].append(incomeTree.predict(X_test[i]))

predictions_array = np.array(list(predictions.values()), dtype=int).T
column_names = list(predictions.keys())
csv_file_path = "output.csv"
np.savetxt(csv_file_path, predictions_array, delimiter=',', header=','.join(column_names), fmt='%d', comments='')