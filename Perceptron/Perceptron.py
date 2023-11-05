import numpy as np
import sys
import math

class Standard_Perceptron:
  def __init__(self, max_epochs=10, learning_rate = 0.1):
    self.max_epochs = max_epochs
    self.learning_rate = learning_rate

  def train(self, X, y):
    num_examples = X.shape[0]
    num_features = X.shape[1]
    weights = np.full(num_features, np.round(1/num_examples, 2), dtype=float)
    for epoch in range (1, self.max_epochs):
      wrong_pred = 0
      # Shuffles the data
      shuffled_indices = self._shuffle(len(y))
      X_shuffled = X[shuffled_indices]
      y_shuffled = y[shuffled_indices]

      for i in shuffled_indices:
        if (y_shuffled[i] * np.dot(weights,X_shuffled[i]) <= 0):
          wrong_pred += 1
          weights = weights + self.learning_rate * y_shuffled[i] * X_shuffled[i]
      print(wrong_pred, 'wrong out of',num_examples)
    self.weights = weights
    
    return weights
  
  def predict(self, X):
    return np.sign(np.dot(self.weights, X))

  def _shuffle(self, len):
    shuffled = np.arange(len)
    np.random.shuffle(shuffled)
    return shuffled
  
class Voted_Perceptron:
  def __init__(self, max_epochs=10, learning_rate = 0.1):
    self.max_epochs = max_epochs
    self.learning_rate = learning_rate

  def train(self, X, y):
    num_examples = X.shape[0]
    num_features = X.shape[1]
    weights = np.full(num_features, np.round(1/num_examples, 2), dtype=float)
    vote = 1
    weight_list = [weights]
    vote_list = []
    # weight_dict = {weights: 0}
    for epoch in range (1, self.max_epochs):
      wrong_pred = 0
      # Shuffles the data
      # shuffled_indices = self._shuffle(len(y))
      # X_shuffled = X[shuffled_indices]
      # y_shuffled = y[shuffled_indices]

      X_shuffled = X
      y_shuffled = y

      # for i in shuffled_indices:
      for i in range(num_examples):
        # print('prediction:',np.dot(weights,X_shuffled[i]), 'where y=',y_shuffled[i])
        if (y_shuffled[i] * np.dot(weights,X_shuffled[i]) <= 0):
          # print('mistake on',y_shuffled[i])
          wrong_pred += 1
          
          weights = weights + self.learning_rate * y_shuffled[i] * X_shuffled[i]
          vote_list.append(vote)
          weight_list.append(weights)
          vote = 1
        else:
          vote += 1
          # print('new weight',weights)
        # print('weights',weights,'vote',vote)
      print(wrong_pred, 'wrong out of',num_examples)
    vote_list.append(vote)
    self.weight_list = weight_list
    self.vote_list = vote_list
    
    result = [(weight_list[i], vote_list[i]) for i in range(len(weight_list))]
    return result
  
  def predict(self, X):
    prediction = 0
    for i in range(len(self.weight_list)):
      prediction += self.vote_list[i] * np.sign(np.dot(self.weight_list[i],X))
      # print('weights',self.weight_list[i], 'vote', self.vote_list[i], 'prediction:',prediction)
    return np.sign(prediction)

  def _shuffle(self, len):
    shuffled = np.arange(len)
    np.random.shuffle(shuffled)
    return shuffled
  