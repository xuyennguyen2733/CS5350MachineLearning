import numpy as np

class NeuralNetwork:
  def __init__(self, network, learning_rate=0.1):
    self.network = network
    self.learning_rate = learning_rate

  def sigmoid(self, x):
    return 1/(1+np.exp(-x))
  
  def sigmoid_derivative(self, x):
    return self.sigmoid(x)*(1-self.sigmoid(x))
  
  def loss(self, y, y_star):
    return 1/2 * (y - y_star) * (y - y_star)
  
  def loss_derivative(self, y, y_star):
    return y - y_star
  
  def activate(self, X, y):
    # calculate the activation values of the first layer:
    first_layer = self.network[0]
    z = self.sigmoid(np.dot(first_layer['weights'],X))
    first_layer['output'] = np.concatenate(([1],z))
    # print(first_layer['output'])

    # calculate the activation values of the hidden layer:
    for i in range(1, len(self.network)-1):
      layer = self.network[i]
      prev_layer = self.network[i-1]
      # print(layer['weights'],prev_layer['output'],np.dot(layer['weights'],prev_layer['output']))
      z = self.sigmoid(np.dot(layer['weights'],prev_layer['output']))
      layer['output'] = np.concatenate(([1],z))
      # print(layer['output'])

    # calculate the output
    last_layer = self.network[len(self.network)-1]
    prev_last_layer = self.network[len(self.network)-2]
    last_layer['output'] = np.dot(last_layer['weights'], prev_last_layer['output'])
    return last_layer['output']



  def backpropagate(self, X, y):
    outputs = np.array([X])

    for i in range(len(self.network)-1):
        layer = self.network[i]
        layer['weighted_sum'] = np.dot(layer['weights'],outputs[-1])
        layer['output'] = self.sigmoid(layer['weighted_sum'])
        layer['output'] = np.concatenate(([1],layer['output']))
        outputs = np.concatenate((outputs,[layer['output']]))

    last_layer = self.network[len(self.network)-1]
    print()
    last_layer['weighted_sum'] = np.dot(last_layer['weights'], outputs[-1])
    last_layer['output'] = np.concatenate(([0,0],[last_layer['weighted_sum']]))
    outputs = np.concatenate((outputs, [last_layer['output']]))
    print('outputs',outputs)

    loss = self.loss(y, outputs[-1])

    print('loss derivative',self.loss_derivative(y, outputs[-1][-1]))
    print('sigmoid derivative',self.sigmoid_derivative(layer['weighted_sum']))
    last_layer['local_gradient'] = self.loss_derivative(y, outputs[-1][-1]) * self.sigmoid_derivative(layer['weighted_sum'])

    for i in reversed(range(1,len(self.network))):
        layer = self.network[i]

        next_layer = self.network[i + 1]
        layer['local_gradient'] = np.dot(next_layer['weights'].values(), next_layer['local_gradient']) * self.sigmoid_derivative(layer['weighted_sum'])

        layer['weight_gradients'] = np.outer(layer['local_gradient'], outputs[i])

        layer['weights'] -= self.learning_rate * layer['weight_gradients']

    return loss
  
class Backpropagate():
  def __init__():
     pass

  # Assuming the neural network is represented as a list of dictionaries
# Each dictionary contains 'weights' (a 2D array), 'output' (an array), etc.

# Function to compute the sigmoid activation function
  def sigmoid(self,x):
      return 1 / (1 + np.exp(-x))

  # Function to compute the derivative of the sigmoid activation function
  def sigmoid_derivative(self,x):
      return self.sigmoid(x) * (1 - self.sigmoid(x))

  # Function to compute the mean squared error loss
  def mean_squared_error(self,y, y_pred):
      return 0.5 * np.mean((y - y_pred)**2)

  # Function to compute the derivative of the mean squared error loss
  def mean_squared_error_derivative(self,y, y_pred):
      return y_pred - y

  # Backpropagation function
  def backpropagate(self,network, X, y, learning_rate):
      # Forward pass
      outputs = [X]
      for layer in network:
          layer['weighted_sum'] = np.dot(outputs[-1], layer['weights'])
          layer['output'] = np.sign(layer['weighted_sum'])  # Activation function, assuming binary output {-1, 1}
          outputs.append(layer['output'])

      # Compute loss
      loss = mean_squared_error(y, outputs[-1])

      # Backward pass
      for i in reversed(range(len(network))):
          layer = network[i]

          # Compute local gradient
          if i == len(network) - 1:
              # Output layer
              layer['local_gradient'] = mean_squared_error_derivative(y, outputs[-1]) * sigmoid_derivative(layer['weighted_sum'])
          else:
              # Hidden layers
              next_layer = network[i + 1]
              layer['local_gradient'] = np.dot(next_layer['weights'].T, next_layer['local_gradient']) * sigmoid_derivative(layer['weighted_sum'])

          # Compute weight gradients
          layer['weight_gradients'] = np.outer(layer['local_gradient'], outputs[i])

          # Update weights
          layer['weights'] -= learning_rate * layer['weight_gradients']

      return loss
  
# class StochasticSubgradient:
#   def __init__(self, schedule, max_epochs=10, learning_rate = 0.1, c = 1, a = 0.01):
#     self.schedule = schedule
#     self.max_epochs = max_epochs
#     self.c = c
#     if learning_rate<=0:
#       raise Exception("learning_rate must be greater than 0")
#     else:
#       self.learning_rate = learning_rate
#     if a<=0:
#       raise Exception("a must be greater than 0")
#     else:
#       self.a = a

#   def train(self, X, y):
#     num_examples = X.shape[0]
#     num_features = X.shape[1]
#     weights = np.zeros(num_features+1)
#     for epoch in range (1, self.max_epochs + 1):
#       wrong_pred = 0
#       # Shuffles the data
#       shuffled_indices = self._shuffle(len(y))
#       X_shuffled = X[shuffled_indices]
#       y_shuffled = y[shuffled_indices]


#       for i in shuffled_indices:
#         if (y_shuffled[i] * np.dot(weights,self._concat(X_shuffled[i],1)) <= 1):
#           wrong_pred += 1
#           weights = weights - self.learning_rate * self._w0(weights,0) + self.learning_rate * self.c * num_examples * y_shuffled[i] * self._concat(X_shuffled[i], 1)
#         else:
#           weights = (1-self.learning_rate) * weights
        
#       self.learning_rate = self.schedule(self.learning_rate, self.a, epoch)
#       # print("epoch:",epoch,"weights:",weights,"learning_rate:",self.learning_rate)
#       # print(wrong_pred, 'wrong out of',num_examples)
#     self.weights = weights
    
#     return weights

#   def predict(self, X):
#     return np.sign(np.dot(self.weights,self._concat(X,1)))

#   def _shuffle(self, len):
#     shuffled = np.arange(len)
#     np.random.shuffle(shuffled)
#     return shuffled
  
#   def _w0(self, vector, bias):
#     vector_copy = vector.copy()
#     vector_copy[-1] = bias
#     return vector_copy
  
#   def _concat(self, vector, bias):
#     concat_vector = np.append(vector, bias)
#     return concat_vector

# class DualSVM:
#   def __init__(self, c=1):
#     self.c = c

#   def train(self, X, y):
#     num_examples = X.shape[0]

#     def objective(alpha):
#       return 0.5 * np.sum(np.outer(alpha * y, alpha * y) * np.dot(X, X.T)) - np.sum(alpha)
    
#     sum_constraint = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}
#     bound_constraint = [(0, self.c) for i in range(0, num_examples)]
#     init_alpha = np.ones(num_examples) * 1e-5

#     solution = minimize(objective, init_alpha, method='SLSQP', bounds = bound_constraint, constraints=sum_constraint)

#     alpha_star = solution.x
#     w_star = np.sum((alpha_star * y)[:, np.newaxis] * X, axis=0)
#     bias = np.mean(y - np.dot(X, w_star))

#     self.alpha_star = alpha_star
#     self.weights = w_star
#     self.bias = bias

#     return self._concat(self.weights, self.bias)
  
#   def predict(self, X):
#     return np.sign(np.dot(self._concat(self.weights, self.bias),self._concat(X,1)))
  
#   def _concat(self, vector, bias):
#     concat_vector = np.append(vector, bias)
#     return concat_vector
  
# class GaussianKernelSVM:
#   def __init__(self, c=1, learning_rate=0.01):
#     self.c = c
#     self.learning_rate = learning_rate

#   def train(self, X, y):
#     num_examples = X.shape[0]

#     K = self._Gaussian_Kernel_Matrix(X, self.learning_rate)
    
#     def objective(alpha):
#       return 0.5 * np.sum(np.outer(alpha * y, alpha * y) * K) - np.sum(alpha)
#     tol = 1e-5
#     sum_constraint = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}
#     bound_constraint = [(0, self.c) for i in range(0, num_examples)]
#     init_alpha = np.ones(num_examples) * 1e-20
#     # init_alpha = np.full(num_examples, self.c/5)
#     # init_alpha = np.zeros(num_examples)
#     # init_alpha = np.random.uniform(0, self.c, num_examples)
#     solution = minimize(objective, init_alpha, method='SLSQP', bounds = bound_constraint, constraints=sum_constraint)
#     alpha_star = solution.x
#     support_indices = np.where(alpha_star > tol)[0]
#     self.alpha_star = alpha_star[support_indices]
#     self.X = X[support_indices]
#     self.y = y[support_indices]
#     self.bias = self._calculate_bias()

#     return self.alpha_star
  
#   def predict(self, X):
#     difference = np.subtract(self.X,X)
#     K = np.exp(np.sum(-difference**2/self.learning_rate,axis=-1))
#     alpha_y = self.alpha_star*self.y
#     return np.sign(np.sum(alpha_y * K) + self.bias)
  
#   def _concat(self, vector, bias):
#     concat_vector = np.append(vector, bias)
#     return concat_vector
  
#   def _Gaussian_Kernel_Matrix(self, X, learning_rate):
#     differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
#     K = np.exp(np.sum(-differences**2/learning_rate, axis=-1))
#     return K
  
#   def _calculate_bias(self):
#     inner_sum = np.sum(
#         self.alpha_star * self.y * np.exp(
#             np.sum(-np.subtract(self.X[:, np.newaxis, :], self.X) ** 2 / self.learning_rate, axis=-1)
#         ),
#         axis=-1
#     )
#     outer_sum = np.sum(self.y - inner_sum)
    
#     return outer_sum / len(self.y)