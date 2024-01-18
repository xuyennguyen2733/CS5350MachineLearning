import numpy as np
from Perceptron import Standard_Perceptron, Voted_Perceptron, Averaged_Perceptron

data_train = np.genfromtxt("bank-note\train.csv", delimiter=",", unpack=True)
data_train = data_train.T
X_train = data_train[:,:-1]
y_train = data_train[:,-1]
y_train[y_train==0] = -1

data_test = np.genfromtxt("bank-note\test.csv", delimiter=",", unpack=True)
data_test = data_test.T
X_test = data_test[:,:-1]
y_test = data_test[:,-1]
y_test[y_test==0] = -1

learning_rate = 0.02
print("learning rate:", learning_rate)
print()
print("********************")
print("a. STANDARD PERCEPTRON")
print("********************")
standard = Standard_Perceptron(10, learning_rate)
standard_weights = standard.train(X_train, y_train)
print("weight vector:",standard_weights)
standard_error = 0
for x, y in zip(X_test, y_test):
  prediction = standard.predict(x)
  if (prediction*y <= 0):
    standard_error+=1/len(y_test)
print("average error: ", standard_error)
print()

print("********************")
print("b. VOTED PERCEPTRON")
print("********************")
voted = Voted_Perceptron(10, learning_rate)
voted_weight_vote = voted.train(X_train, y_train)
voted_error = 0
print()
print("List of weight vector and count:")
print()
print(f"{'Weight Vector':<40} \t{'Vote'}")
count = 1
for i in range(len(voted_weight_vote)):
    weight = voted_weight_vote[i][0]
    vote = voted_weight_vote[i][1]
    formatted_weight = np.array2string(weight, precision=6, separator=' ', suppress_small=True)
    print(f"{formatted_weight:<40} \t{vote}")
    # print(f"{count} & ${weight}$ & {vote} \\\\")
    count += 1
for x, y in zip(X_test, y_test):
  prediction = voted.predict(x)
  if (prediction*y <= 0):
    voted_error+=1/len(y_test)
print()
print("average error: ", voted_error)
print()

print("********************")
print("c. AVERAGED PERCEPTRON")
print("********************")
averaged = Averaged_Perceptron(10, learning_rate)
averaged_weights = averaged.train(X_train, y_train)
print("weight vector:",averaged_weights)
averaged_error = 0
for x, y in zip(X_test, y_test):
  prediction = averaged.predict(x)
  if (prediction*y <= 0):
    averaged_error+=1/len(y_test)
print("average error: ", averaged_error)
print()