class Node:
  def __init__(self, attribute, parent=None):
    self.attribute = attribute
    self.parent = parent
    self.children = {}

  def addChild(self, childAttribute, childNode):
    self.children[childAttribute] = child_node

  def next(self, childAttribute):
    return self.children[childAttribute]