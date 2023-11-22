import numpy as np
from svm import StochasticSubgradient

data_train = np.genfromtxt("bank-note\\train.csv", delimiter=",", unpack=True)
data_train = data_train.T
X_train = data_train[:,:-1]
y_train = data_train[:,-1]
y_train[y_train==0] = -1

data_test = np.genfromtxt("bank-note\\test.csv", delimiter=",", unpack=True)
data_test = data_test.T
X_test = data_test[:,:-1]
y_test = data_test[:,-1]
y_test[y_test==0] = -1

learning_rate = 0.1
a = 0.01
c_arr = [100/873, 500/873, 700/873]
c_strs = ["100/873", "500/873", "700/873"]

def schedule_1(learning_rate, a, epoch):
  return learning_rate / (1+learning_rate/a * epoch)

def schedule_2(learning_rate, a, epoch):
  return learning_rate / (1+epoch)

subgradient_model_params_1 = []
subgradient_model_params_2 = []
subgradient_train_error_1 = []
subgradient_train_error_2 = []
subgradient_test_error_1 = []
subgradient_test_error_2 = []

print("***************************")
print("2a. STOCHASTIC_SUBGRADIENT: schedule of learning with a decay rate")
print("***************************")
print()
print(f'{"C":<20} \t{"Training Error":<20} \t{"Testing Error"}')
for c, c_str in zip(c_arr, c_strs):
  subgradient = StochasticSubgradient(schedule_1, 100, learning_rate, c, a)
  subgradient_model_params_1.append(subgradient.train(X_train, y_train))
  subgradient_test_error = 0
  subgradient_train_error = 0
  for x, y in zip(X_train, y_train):
    prediction = subgradient.predict(x)
    if (prediction*y <= 0):
      subgradient_train_error+=1/len(y_train)
  for x, y in zip(X_test, y_test):
    prediction = subgradient.predict(x)
    if (prediction*y <= 0):
      subgradient_test_error+=1/len(y_test)
  subgradient_train_error_1.append(subgradient_train_error)
  subgradient_test_error_1.append(subgradient_test_error)
  print(f"{c_str:<20} \t{subgradient_train_error:<20.4f} \t{subgradient_test_error:.4f}")
  # print(f"{count} & ${weight}$ & {vote} \\\\")

print()

print("***************************")
print("2b. STOCHASTIC_SUBGRADIENT: schedule of learning without a decay rate")
print("***************************")
print()
print(f'{"C":<20} \t{"Training Error":<20} \t{"Testing Error"}')
for c, c_str in zip(c_arr, c_strs):
  subgradient = StochasticSubgradient(schedule_2, 100, learning_rate, c, a)
  subgradient_model_params_2.append(subgradient.train(X_train, y_train))
  subgradient_test_error = 0
  subgradient_train_error = 0
  for x, y in zip(X_train, y_train):
    prediction = subgradient.predict(x)
    if (prediction*y <= 0):
      subgradient_train_error+=1/len(y_train)
  for x, y in zip(X_test, y_test):
    prediction = subgradient.predict(x)
    if (prediction*y <= 0):
      subgradient_test_error+=1/len(y_test)
  subgradient_train_error_2.append(subgradient_train_error)
  subgradient_test_error_2.append(subgradient_test_error)
  print(f"{c_str:<20} \t{subgradient_train_error:<20.4f} \t{subgradient_test_error:.4f}")
  # print(f"{count} & ${weight}$ & {vote} \\\\")

print()

print("***************************")
print("2c. STOCHASTIC_SUBGRADIENT: model parameters")
print("***************************")
print()
print(f'{"C":<20} \t{"Model Parameter 1":<40} \t{"Model Parameter 2:":<40} \t{"Difference"}')
for i in range(0,len(c_strs)):
  c_str = c_strs[i]
  error1 = np.array2string(subgradient_model_params_1[i], precision=2, separator=' ', suppress_small=True)
  error2 = np.array2string(subgradient_model_params_2[i], precision=2, separator=' ', suppress_small=True)
  diff = np.array2string(abs(subgradient_model_params_1[i]-subgradient_model_params_2[i]), precision=2, separator=' ', suppress_small=True)
  print(f"{c_str:<20} \t{error1:<40} \t{error2:<40} \t{diff}")

print()
print(f'{"C":<20} \t{"Training Error 1":<40} \t{"Training Error 2:":<40} \t{"Difference"}')
for i in range(0,len(c_strs)):
  c_str = c_strs[i]
  error1 = subgradient_train_error_1[i]
  error2 = subgradient_train_error_2[i]
  print(f"{c_str:<20} \t{error1:<40.4f} \t{error2:<40.4f} \t{abs(error1-error2):.4f}")

print()
print(f'{"C":<20} \t{"Testing Error 1":<40} \t{"Testing Error 2:":<40} \t{"Difference"}')
for i in range(0,len(c_strs)):
  c_str = c_strs[i]
  error1 = subgradient_test_error_1[i]
  error2 = subgradient_test_error_2[i]
  print(f"{c_str:<20} \t{error1:<40.4f} \t{error2:<40.4f} \t{abs(error1-error2):.4f}")




# print("learning rate:", learning_rate)
# print()
# print("********************")
# print("a. STANDARD PERCEPTRON")
# print("********************")
# standard = Standard_Perceptron(10, learning_rate)
# standard_weights = standard.train(X_train, y_train)
# print("weight vector:",standard_weights)
# standard_error = 0
# for x, y in zip(X_test, y_test):
#   prediction = standard.predict(x)
#   if (prediction*y <= 0):
#     standard_error+=1/len(y_test)
# print("average error: ", standard_error)
# print()

# print("********************")
# print("b. VOTED PERCEPTRON")
# print("********************")
# voted = Voted_Perceptron(10, learning_rate)
# voted_weight_vote = voted.train(X_train, y_train)
# voted_error = 0
# print()
# print("List of weight vector and count:")
# print()
# print(f"{'Weight Vector':<40} \t{'Vote'}")
# count = 1
# for i in range(len(voted_weight_vote)):
#     weight = voted_weight_vote[i][0]
#     vote = voted_weight_vote[i][1]
#     formatted_weight = np.array2string(weight, precision=6, separator=' ', suppress_small=True)
#     print(f"{formatted_weight:<40} \t{vote}")
#     # print(f"{count} & ${weight}$ & {vote} \\\\")
#     count += 1
# for x, y in zip(X_test, y_test):
#   prediction = voted.predict(x)
#   if (prediction*y <= 0):
#     voted_error+=1/len(y_test)
# print()
# print("average error: ", voted_error)
# print()

# print("********************")
# print("c. AVERAGED PERCEPTRON")
# print("********************")
# averaged = Averaged_Perceptron(10, learning_rate)
# averaged_weights = averaged.train(X_train, y_train)
# print("weight vector:",averaged_weights)
# averaged_error = 0
# for x, y in zip(X_test, y_test):
#   prediction = averaged.predict(x)
#   if (prediction*y <= 0):
#     averaged_error+=1/len(y_test)
# print("average error: ", averaged_error)
# print()