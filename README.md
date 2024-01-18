DECISION TREE

The DecisionTree class constructor takes in 4 parameters:

1. possibleValues (mandatory): a dictionary whose keys are the features, and the value of each key is a list of all possible values under each feature
2. maxDepth (optional): defaulted to 'None', which means there is not limit in depth and the tree will grow until the data split reach purity. Otherwise, maxDepth is an int specifying the maximum depth of the tree.
3. criterion (optional): defaulted to 'entropy', a string that specifies the method for splitting data. Accepted values include 'entropy', 'gini_index', and 'majority_error'.
4. unknownIsMissing (optional): defaulted to 'False', specifies whether to treat 'unknown' values as a type of value under certain feature or as a missing value.

HOW TO LEARN A TREE

First, initialize a tree object. For example:

possibleValues = {
'name': ['John', 'Patty', 'Phuong'],
'major': ['CS', 'Game', 'Music', 'Education', 'Politics]
}
myTree = DecisionTree(possibleValue)

Second, process the data:

1. Make sure there are 2 variables:
   X - an n by m array, where n is the number of features, and m is the number of input examples. X must have headers being the feature names.
   Y - an 1 by m array, where m is the number of output labels. Y must have a header for consistency purposes.
   It's best to use numpy arrays to construct X and Y

For example: obtaining the X_train and Y_train data in the desire format

    column_headers = ['buying','maint','doors','persons','lug_boot','safety','label']
    possibleValues = {
    'buying': ['vhigh', 'high', 'med', 'low'],
    'maint': ['vhigh', 'high', 'med', 'low'],
    'doors': ['2', '3', '4', '5more'],
    'persons': ['2', '4', 'more'],
    'lug_boot': ['small', 'med', 'big'],
    'safety': ['low', 'med', 'high'],
    }

    data = np.genfromtxt(".\\car-4\\train.csv", dtype=None, delimiter=",", names=column_headers, encoding=None)

    x_labels = []
    num_features = len(column_headers)-1
    for i in range(0,num_features):
    x_labels.append( column_headers[i])

    y_label = column_headers[-1]

    X_train = data[x_labels]
    y_train = data[y_label]

2. Start learning a tree from the processed data:

   root = myTree.ID3(X_train, y_train)

3. Predict using an input:

   prediction = myTree.predict(X_test[index])
   print(prediction)

   \*\*\* Make sure X_test and Y_test also has the same format as X_train and Y_train

ADABOOST (unable to correctly implement)

Make sure to transform the output values to {-1,1} if they are not in that format already.
For example, the data set for bank marketing has output 'no' and 'yes'. Map these values to -1 and 1 as follow:
map = {'no': -1, 'yes': 1}
y_train = np.vectorize(map.get)(y)
Then use y_train as normal

BAGGING (unable to implement)

RANDOM FOREST (unable to implement)

PERCEPTRON

1. Data processing:
   Make sure your data meet the following requirements:

   - All values are numerical
   - Lables are 1 or -1
   - data.shape must return (m,n) where m is the number of examples and n is the number of features
   - Input data X and labels y must be storeD in two separate variables

2. Training the model:

   model = STANDARD_PERCEPTRON(epoch_num, learning_rate) (epoch_num is defaulted to 10, while learning rate is defaulted to 0.1)
   weight = model.train(X,y)

Can replace STANDARD_PERCEPTRON with VOTED_PERCEPTRON or AVERAGED_PERCEPTRON
For VOTED_PERCEPTRON, the returned value isn't a single weight vector but a list of tuples of weight vectors and their corresponding votes

3. Predict using an input:

   prediction = model.predict(x) (x is a single input vector of length n, i.e. n features)

SVM

1. Data processing:
   Make sure your data meet the following requirements:

   - All values are numerical
   - Lables are 1 or -1
   - data.shape must return (m,n) where m is the number of examples and n is the number of features
   - Input data X and labels y must be stored in two separate variables

   For StochasticSubgradient(), prepare a learning schedule function that takes in 3 parameters: learning_rate, decay_rate, and epoch respectively, and returns the new learning rate.
   Note: include the decay_rate as the second parameter of the function regardless of whether or not the formula uses one.

2. Training the model:

   subgradient_model = StochasticSubgradient(schedule, max_epochs, learning_rate, c, a) (default values: max_epochs=10, learning_rate = 0.1, c = 1, a = 0.01)
   subgradient_model.train(X,y) (return weights)

   dual_model = DualSVM(c) (default value: c = 1)
   dual_model.train(X,y) (returns [weights, bias])

   gaussian_model = GaussianKernelSVM(c, learning_rate) (default values: c=1, learning_rate=0.01)
   gaussian_model.train(X,y) (returns alpha\*)

3. Predict using an input:

   prediction = model.predict(x) (x is a single input vector of length n, i.e. n features)
