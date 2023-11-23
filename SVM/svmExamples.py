import numpy as np
from SVM import StochasticSubgradient, DualSVM, GaussianKernelSVM

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
print("2c. STOCHASTIC_SUBGRADIENT: Comparison")
print("***************************")
print()
print(f'{"C":<20} \t{"[w, bias] 1":<40} \t{"[w, bias] 2:":<40} \t{"Difference"}')
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
print()

print("***************************")
print("3a. DUAL SVM")
print("***************************")
print()
feature_weights = []
bias = []
print(f'{"C":<10} \t{"Feature Weights":<50} \t{"Bias"} \t{"Testing Error"}')
for c, c_str in zip(c_arr, c_strs):
  test_error = 0
  svm = DualSVM(c)
  svm.train(X_train, y_train)
  feature_weights.append(svm.weights)
  bias.append(svm.bias)
  formatted_weights = np.array2string(svm.weights, separator=' ', suppress_small=True, formatter={'float_kind': '{: .4e}'.format})
  # print(svm.weights)
  for i in range (0, len(y_test)):
    prediction = svm.predict(X_test[i])
    if (prediction*y_test[i] <= 0):
      test_error+=1/len(y_test)
  print(f'{c_str:<10} \t{formatted_weights:<50} \t{svm.bias:.4f} \t{test_error:.4f}')
print()

print("***************************")
print("3b. DUAL SVM with GAUSSIAN KERNEL")
print("***************************")
print()
print("Testing and Training Errors for Each C and Gamma Settings:")
print()
learning_rates = [0.1, 0.5, 1, 5, 100]
alpha_vectors = {}
print(f'{"Learning Rate":<10} \t{"C = {}".format(c_strs[0]):<30} \t{"C = {}".format(c_strs[1]):<30} \t{"C = {}".format(c_strs[2]):<10}')
for learning_rate in learning_rates:
  kernel_train_error = []
  kernel_test_error = []
  for c, c_str in zip(c_arr,c_strs):
    test_error = 0
    train_error = 0
    svm = GaussianKernelSVM(c,learning_rate)
    svm.train(X_train, y_train)
    alpha_vectors[(learning_rate, c_str)] = svm.alpha_star
    for i in range (0, len(y_test)):
      prediction = svm.predict(X_test[i])
      if (prediction*y_test[i] <= 0):
        test_error+=1/len(y_test)
    for i in range (0, len(y_train)):
      prediction = svm.predict(X_train[i])
      if (prediction*y_train[i] <= 0):
        train_error+=1/len(y_train)
    kernel_train_error.append(train_error)
    kernel_test_error.append(test_error)
  print(f'{learning_rate:<10} \t{"train = {:.4f}, test = {:.4f}".format(kernel_train_error[0], kernel_test_error[0]):<30} \t{"train = {:.4f}, test = {:.4f}".format(kernel_train_error[1], kernel_test_error[1]):<30} \t{"train = {:.4f}, test = {:.4f}".format(kernel_train_error[2], kernel_test_error[2]):<30}')
print()

print("***************************")
print("3c. SUPPORT VECTORS")
print("***************************")
print()
print('Number of support vectors:')
print()
report_sv_count_head = f'{"C":<10}'
for learning_rate in learning_rates:
  report_sv_count_head += f'\t{"Learning Rate = {}".format(learning_rate):<20}'
print(report_sv_count_head)
tol = 1e-5
common_sv_count = {}
prev_sv_indices = None
for c_str in c_strs:
  report_sv_count = f'{c_str:<10}'
  for learning_rate in learning_rates:
    alpha = alpha_vectors[(learning_rate,c_str)]
    sv_indices = np.where(alpha > tol)[0]
    if (prev_sv_indices is not None):
      common_sv_count[(learning_rate, c_str)] = len(np.intersect1d(sv_indices, prev_sv_indices))
    prev_sv_indices = sv_indices
    sv_count = len(sv_indices)
    report_sv_count += f'\t{"{}".format(sv_count):<20}'
  print(report_sv_count)

print()
report_common_sv_head = f'{"C":<10} \t{learning_rates[0]} - {learning_rates[1]:<5}'
for i in range(1, len(learning_rates)-1):
  report_common_sv_head += f'\t{learning_rates[i]} - {learning_rates[i+1]:<5}'
print("Number of common support vectors between consecutive learning rates:")
print()
print(report_common_sv_head)
for c_str in c_strs:
  report = f'{c_str:<10}'
  for learning_rate in learning_rates[1:]:
    report += f'\t{common_sv_count[(learning_rate, c_str)]:<10}'
  print(report)


# # *************************
# # TEST SECTION 
# # *************************
# def predict(alpha, X, y, bias, x_test):
#   print('PREDICTION')
#   difference = -np.subtract(x_test,X)
#   print('x_i - x', difference)
#   K = np.exp(np.sum(difference**2/learning_rate,axis=-1))
#   alpha_y = alpha*y
#   print("K(xi,X)",K)
#   print("a_i * y_i", alpha_y)
#   print("a_i * y_i * K(x_i,X)", alpha_y * K)
#   print("sum", np.sum(alpha_y * K))
#   return np.sign(np.sum(alpha_y * K) + bias)

# def objective(alpha, X, y, learning_rate):
#   print('OBJECTIVE')
#   K = _Gaussian_Kernel_Matrix(X, learning_rate)
#   print('K',K)
#   return 0.5 * np.sum(np.outer(alpha * y, alpha * y) * K) - np.sum(alpha)
# def _Gaussian_Kernel_Matrix(X, learning_rate):
#   differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
#   print('x_i - x_j', differences)
#   K = np.exp(np.sum(-differences**2/learning_rate, axis=-1))
#   return K
# def calculate_bias(X, y, learning_rate):
#   K_support = _Gaussian_Kernel_Matrix(X, learning_rate)
#   bias = np.mean(y - np.dot(K_support.T, alpha * y))
#   return bias


# X=np.array([[1,1],[2,2],[3,3],[4,4],[5,5]])
# y = np.array([-1,1,-1,1,1])
# alpha = np.full(len(y),0.01)
# bias = 0.05
# x_test = np.array([1,1])
# print(predict(alpha, X, y, bias, x_test))
# print(objective(alpha, X, y, learning_rate))