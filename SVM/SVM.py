import numpy as np
from scipy.optimize import minimize

class StochasticSubgradient:
  def __init__(self, schedule, max_epochs=10, learning_rate = 0.1, c = 1, a = 0.01):
    self.schedule = schedule
    self.max_epochs = max_epochs
    self.c = c
    if learning_rate<=0:
      raise Exception("learning_rate must be greater than 0")
    else:
      self.learning_rate = learning_rate
    if a<=0:
      raise Exception("a must be greater than 0")
    else:
      self.a = a

  def train(self, X, y):
    num_examples = X.shape[0]
    num_features = X.shape[1]
    weights = np.zeros(num_features+1)
    for epoch in range (1, self.max_epochs + 1):
      wrong_pred = 0
      # Shuffles the data
      shuffled_indices = self._shuffle(len(y))
      X_shuffled = X[shuffled_indices]
      y_shuffled = y[shuffled_indices]


      for i in shuffled_indices:
        if (y_shuffled[i] * np.dot(weights,self._concat(X_shuffled[i],1)) <= 1):
          wrong_pred += 1
          weights = weights - self.learning_rate * self._w0(weights,0) + self.learning_rate * self.c * num_examples * y_shuffled[i] * self._concat(X_shuffled[i], 1)
        else:
          weights = (1-self.learning_rate) * weights
        
      self.learning_rate = self.schedule(self.learning_rate, self.a, epoch)
      # print("epoch:",epoch,"weights:",weights,"learning_rate:",self.learning_rate)
      # print(wrong_pred, 'wrong out of',num_examples)
    self.weights = weights
    
    return weights

  def predict(self, X):
    return np.sign(np.dot(self.weights,self._concat(X,1)))

  def _shuffle(self, len):
    shuffled = np.arange(len)
    np.random.shuffle(shuffled)
    return shuffled
  
  def _w0(self, vector, bias):
    vector_copy = vector.copy()
    vector_copy[-1] = bias
    return vector_copy
  
  def _concat(self, vector, bias):
    concat_vector = np.append(vector, bias)
    return concat_vector

class DualSVM:
  def __init__(self, c=1):
    self.c = c

  def train(self, X, y):
    num_examples = X.shape[0]

    def objective(alpha):
      return 0.5 * np.sum(np.outer(alpha * y, alpha * y) * np.dot(X, X.T)) - np.sum(alpha)
    
    sum_constraint = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}
    bound_constraint = [(0, self.c) for i in range(0, num_examples)]
    init_alpha = np.ones(num_examples) * 1e-5

    solution = minimize(objective, init_alpha, method='SLSQP', bounds = bound_constraint, constraints=sum_constraint)

    alpha_star = solution.x
    w_star = np.sum(np.dot(alpha_star, y) * X, axis=0)
    bias = np.mean(y - np.dot(X, w_star))

    self.alpha_star = alpha_star
    self.weights = w_star
    self.bias = bias

    return self._concat(self.weights, self.bias)
  
  def predict(self, X):
    return np.sign(np.dot(self._concat(self.weights, self.bias),self._concat(X,1)))
  
  def _concat(self, vector, bias):
    concat_vector = np.append(vector, bias)
    return concat_vector
  
class GaussianKernelSVM:
  def __init__(self, c=1, learning_rate=0.01):
    self.c = c
    self.learning_rate = learning_rate

  def train(self, X, y):
    num_examples = X.shape[0]

    K = self._Gaussian_Kernel_Matrix(X, self.learning_rate)
    
    def objective(alpha):
      return 0.5 * np.sum(np.outer(alpha * y, alpha * y) * K) - np.sum(alpha)
    
    sum_constraint = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}
    bound_constraint = [(0, self.c) for i in range(0, num_examples)]
    init_alpha = np.ones(num_examples) * 1e-20
    # init_alpha = np.full(num_examples, self.c/5)
    # init_alpha = np.zeros(num_examples)
    # init_alpha = np.random.uniform(0, self.c, num_examples)
    solution = minimize(objective, init_alpha, method='SLSQP', bounds = bound_constraint, constraints=sum_constraint)
    alpha_star = solution.x
    w_star = np.sum(np.dot(alpha_star, y) * X, axis=0)
    bias = np.mean(y - np.dot(X, w_star))

    self.alpha_star = alpha_star
    self.weights = w_star
    self.bias = bias
    self.X = X
    self.y = y

    return self._concat(self.weights, self.bias)
  
  def predict(self, X):
    K = np.sum(np.exp(np.sum(-np.subtract(X,self.X)**2/self.learning_rate,axis=-1)))
    alpha_y = np.dot(self.alpha_star,self.y)
    return np.sign(alpha_y * K + self.bias)
  
  def _concat(self, vector, bias):
    concat_vector = np.append(vector, bias)
    return concat_vector
  
  def _Gaussian_Kernel_Matrix(self, X, learning_rate):
    differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    K = np.exp(np.sum(-differences**2/learning_rate, axis=-1))
    return K