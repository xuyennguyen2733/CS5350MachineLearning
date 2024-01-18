class Node:
  def __init__(self, attribute):
    self.attribute = attribute
    self.children = None

  def addChild(self, edgeLabel, childNode):
    if self.children == None:
      self.children = {}
    self.children[edgeLabel] = childNode

  def next(self, childAttribute):
    return self.children[childAttribute]
  
  """
  Node root = Node('Outlook')
  Node Outlook = Node('Humidity')
  root.addChild('s', Outlook)
  Node Humidity = Node('Humidity')
  root.addChild('h', Humidity)
  root.addChild('o', None)
  """

  """
  input = {
    'Outlook': o
    'Humidity': h
    'temperature': c
  }
  def predict(root, input):
    splitAt = root.attribute
    splitLabel = input[splitAt]
    currentNode = root.children[splitLabel]
    while (currentNode != None):
     splitAt = currentNode.attribute
     splitLabel = input[splitAt]
     currentNode = currentNode.children[splitLabel]

    return splitLabel
  """