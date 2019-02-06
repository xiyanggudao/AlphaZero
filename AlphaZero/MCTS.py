import numpy as np


class MCTEdge:

	def __init__(self, parentNode, action, P):
		self.W = 0
		self.N = 0
		self.Q = 0
		self.P = P
		self.action = action
		self.parentNode = parentNode
		self.childNode = None

	def setChildNode(self, childNode):
		self.childNode = childNode

	def backup(self):
		if not self.childNode:
			return
		self.N += 1
		self.W += self.childNode.v
		self.Q = self.W / self.N


class MCTNode:

	def __init__(self, actions, P, v, parentEdge = None):
		self.v = v
		self.parentEdge = parentEdge
		self.edges = []
		for i in range(len(P)):
			self.edges[i] = MCTEdge(self, actions[i], P[i])

	def U(self, Cpuct):
		sumN = 0
		for edge in self.edges:
			sumN += edge.N
		sqrtSumN = sumN ** 0.5
		returnValue = np.empty(len(self.edges), np.float32)
		for i in range(len(self.edges)):
			returnValue[i] = Cpuct * self.edges[i].P * sqrtSumN / (1 + self.edges[i].N)
		return returnValue

	def Pi(self, temperature):
		sumN = 0
		exp = 1 / temperature
		for edge in self.edges:
			sumN += edge.N ** exp
		returnValue = np.empty(len(self.edges), np.float32)
		for i in range(len(self.edges)):
			returnValue[i] = self.edges[i].N ** exp / sumN
		return returnValue

	def select(self, Cpuct, eps = 1E-8):
		maxActionValue = 0
		maxActionValueEdge = None
		U = self.U(Cpuct)
		for i in range(len(self.edges)):
			actionValue = U[i] + self.edges[i].Q
			if maxActionValue + eps < actionValue:
				maxActionValue = actionValue
				maxActionValueEdge = self.edges[i]
		return maxActionValueEdge


class MCTS:

	def __init__(self, game, Cpuct=0.5, temperature=1):
		self.game = game
		self.nodeCount = 0
		self.rootNode = None
		self.Cpuct = Cpuct
		self.temperature = temperature

	def getNodeCount(self):
		return self.nodeCount

	def backup(self, edge):
		while edge:
			edge.backup()
			edge = edge.parentNode.parentEdge

	def createNewNode(self, parentEdge = None):
		P, v = self.game.getEvaluation()
		actions = self.game.getActions()
		newNode = MCTNode(actions, P, v, parentEdge)
		self.nodeCount += 1
		return newNode

	def expandNode(self, node):
		edge = node.select(self.Cpuct)
		if not edge:
			return None
		self.game.takeAction(edge.action)
		if edge.childNode: # recursion
			newNode = self.expandNode(edge.childNode)
		else: # create new node here
			newNode = self.createNewNode(edge)
			edge.childNode = newNode
		self.game.undoAction()
		return newNode

	def expand(self):
		if self.rootNode:
			newNode = self.expandNode(self.root, self.game)
		else:
			self.rootNode = self.createNewNode()
			newNode = self.rootNode
		if newNode:
			self.backup(newNode.parentEdge)
			return True
		return False

	def Pi(self):
		if self.rootNode:
			return self.rootNode(self.temperature)
		else:
			return None

	def play(self, actionIndex):
		if not self.rootNode or actionIndex < 0 or actionIndex > len(self.rootNode.edges):
			return None
		edge = self.rootNode.edges[actionIndex]
		self.game.takeAction(edge.action)
		self.rootNode = edge.childNode
		self.rootNode.parentEdge = None # release other old branches
		self.nodeCount = edge.N
		return edge.action
