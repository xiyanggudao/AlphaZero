import numpy as np
import sys


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
		self.edgeSize = len(P)
		self.edges = {} # a dict instead of list, to save memory
		for i in range(self.edgeSize):
			if P[i] > 0:
				self.edges[i] = MCTEdge(self, actions[i], P[i])

	def U(self, Cpuct):
		sumN = 0
		for edge in self.edges.values():
			sumN += edge.N
		sqrtSumN = sumN ** 0.5
		returnValue = np.zeros(self.edgeSize, np.float32)
		for i in self.edges:
			returnValue[i] = Cpuct * self.edges[i].P * sqrtSumN / (1 + self.edges[i].N)
		return returnValue

	def Pi(self, temperature):
		sumN = 0
		exp = 1 / temperature
		for edge in self.edges.values():
			sumN += edge.N ** exp
		assert sumN != 0
		returnValue = np.zeros(self.edgeSize, np.float32)
		for i in self.edges:
			returnValue[i] = self.edges[i].N ** exp / sumN
		return returnValue

	def select(self, Cpuct, eps = 1E-8):
		maxActionValue = -1
		maxActionValueEdge = None
		U = self.U(Cpuct)
		for i in self.edges:
			actionValue = U[i] + self.edges[i].Q
			if maxActionValue + eps < actionValue:
				maxActionValue = actionValue
				maxActionValueEdge = self.edges[i]
		return maxActionValueEdge


class MCTSConfig:

	def __init__(self):
		self.Cpuct = 1.5
		self.temperature = 1
		self.maxNodes = 2**10


class MCTS:

	def __init__(self, game, config):
		self.game = game
		self.nodeCount = 0
		self.rootNode = None
		self.config = config

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
		edge = node.select(self.config.Cpuct)
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
			newNode = self.expandNode(self.rootNode)
		else:
			self.rootNode = self.createNewNode()
			newNode = self.rootNode
		if newNode:
			self.backup(newNode.parentEdge)
			return True
		return False

	def expandMaxNodes(self):
		print('.', end='')
		sys.stdout.flush()
		newNodeCount = 0
		while self.getNodeCount() < self.config.maxNodes:
			if not self.expand():
				break
			newNodeCount += 1
		return newNodeCount

	def Pi(self):
		if self.rootNode:
			return self.rootNode.Pi(self.config.temperature)
		else:
			return None

	def play(self, actionIndex):
		if not self.rootNode or actionIndex not in self.rootNode.edges:
			return None
		edge = self.rootNode.edges[actionIndex]
		self.game.takeAction(edge.action)
		self.rootNode = edge.childNode
		if self.rootNode:
			self.rootNode.parentEdge = None # release other old branches
		self.nodeCount = edge.N
		return edge.action

	def reset(self):
		self.rootNode = None
		self.nodeCount = 0
		self.game.reset()
