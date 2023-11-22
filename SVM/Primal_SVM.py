import numpy as np

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

# class Standard_Perceptron:
#   def __init__(self, max_epochs=10, learning_rate = 0.1):
#     self.max_epochs = max_epochs
#     self.learning_rate = learning_rate

#   def train(self, X, y):
#     num_examples = X.shape[0]
#     num_features = X.shape[1]
#     weights = np.zeros(num_features)
#     for epoch in range (1, self.max_epochs):
#       wrong_pred = 0
#       # Shuffles the data
#       shuffled_indices = self._shuffle(len(y))
#       X_shuffled = X[shuffled_indices]
#       y_shuffled = y[shuffled_indices]

#       for i in shuffled_indices:
#         if (y_shuffled[i] * np.dot(weights,X_shuffled[i]) <= 0):
#           wrong_pred += 1
#           weights = weights + self.learning_rate * y_shuffled[i] * X_shuffled[i]
#       # print(wrong_pred, 'wrong out of',num_examples)
#     self.weights = weights
    
#     return weights
  
#   def predict(self, X):
#     return np.sign(np.dot(self.weights, X))

#   def _shuffle(self, len):
#     shuffled = np.arange(len)
#     np.random.shuffle(shuffled)
#     return shuffled
  
# class Voted_Perceptron:
#   def __init__(self, max_epochs=10, learning_rate = 0.1):
#     self.max_epochs = max_epochs
#     self.learning_rate = learning_rate

#   def train(self, X, y):
#     num_examples = X.shape[0]
#     num_features = X.shape[1]
#     weights = np.zeros(num_features)
#     vote = 1
#     weight_list = [weights]
#     vote_list = []
#     # weight_dict = {weights: 0}
#     for epoch in range (1, self.max_epochs):
#       wrong_pred = 0
#       # Shuffles the data
#       # shuffled_indices = self._shuffle(len(y))
#       # X_shuffled = X[shuffled_indices]
#       # y_shuffled = y[shuffled_indices]

#       X_shuffled = X
#       y_shuffled = y

#       # for i in shuffled_indices:
#       for i in range(num_examples):
#         # print('prediction:',np.dot(weights,X_shuffled[i]), 'where y=',y_shuffled[i])
#         if (y_shuffled[i] * np.dot(weights,X_shuffled[i]) <= 0):
#           # print('mistake on',y_shuffled[i])
#           wrong_pred += 1
          
#           weights = weights + self.learning_rate * y_shuffled[i] * X_shuffled[i]
#           vote_list.append(vote)
#           weight_list.append(weights)
#           vote = 1
#         else:
#           vote += 1
#           # print('new weight',weights)
#         # print('weights',weights,'vote',vote)
#       # print(wrong_pred, 'wrong out of',num_examples)
#     vote_list.append(vote)
#     self.weight_list = weight_list
#     self.vote_list = vote_list
    
#     result = [(weight_list[i], vote_list[i]) for i in range(len(weight_list))]
#     return result
  
#   def predict(self, X):
#     prediction = 0
#     for i in range(len(self.weight_list)):
#       prediction += self.vote_list[i] * np.sign(np.dot(self.weight_list[i],X))
#       # print('weights',self.weight_list[i], 'vote', self.vote_list[i], 'prediction:',prediction)
#     return np.sign(prediction)

#   def _shuffle(self, len):
#     shuffled = np.arange(len)
#     np.random.shuffle(shuffled)
#     return shuffled
  
# class Averaged_Perceptron:
#   def __init__(self, max_epochs=10, learning_rate = 0.1):
#     self.max_epochs = max_epochs
#     self.learning_rate = learning_rate

#   def train(self, X, y):
#     num_examples = X.shape[0]
#     num_features = X.shape[1]
#     weights = np.zeros(num_features)
#     average = np.zeros(num_features)
#     for epoch in range (1, self.max_epochs):
#       wrong_pred = 0
#       # # Shuffles the data
#       # shuffled_indices = self._shuffle(len(y))
#       # X_shuffled = X[shuffled_indices]
#       # y_shuffled = y[shuffled_indices]

#       X_shuffled = X
#       y_shuffled = y

#       # for i in shuffled_indices:
#       for i in range(num_examples):
#         # print('prediction:',np.dot(weights,X_shuffled[i]), 'where y=',y_shuffled[i])
#         if (y_shuffled[i] * np.dot(weights,X_shuffled[i]) <= 0):
#           # print('mistake on',y_shuffled[i])
#           wrong_pred += 1
#           weights = weights + self.learning_rate * y_shuffled[i] * X_shuffled[i]
#         average += weights
#           # print('new weight',weights)
#         # print('weights',weights,'vote',vote)
#       # print(wrong_pred, 'wrong out of',num_examples)
#     self.average = average
#     return average
  
#   def predict(self, X):
#     return np.sign(np.dot(self.average,X))

#   def _shuffle(self, len):
#     shuffled = np.arange(len)
#     np.random.shuffle(shuffled)
#     return shuffled
  