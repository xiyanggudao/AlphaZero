import numpy as np
import sys
from AlphaZero.Game import Game


class MCTEdge:

	def __init__(self, parentNode, index, action, P):
		self.W = 0
		self.N = 0
		self.Q = 0
		self.P = P
		self.action = action
		self.index = index
		self.parentNode = parentNode
		self.childNode = None

	def setChildNode(self, childNode):
		self.childNode = childNode

	def backup(self, v):
		self.N += 1
		self.W += v
		self.Q = self.W / self.N


class MCTNode:

	def __init__(self, actions, P, v, parentEdge = None):
		self.v = v
		self.parentEdge = parentEdge
		self.edgeSize = len(P)
		self.edges = []
		for i in range(self.edgeSize):
			if P[i] > 0:
				self.edges.append(MCTEdge(self, i, actions[i], P[i]))

	def getEdge(self, actionIndex):
		for edge in self.edges:
			if edge.index == actionIndex:
				return edge
		return None

	def U(self, Cpuct):
		sumN = 1E-8 # eps is better than 0 when maxNodes is low
		for edge in self.edges:
			sumN += edge.N
		sqrtSumN = sumN ** 0.5
		returnValue = np.zeros(self.edgeSize, np.float32)
		for edge in self.edges:
			returnValue[edge.index] = Cpuct * edge.P * sqrtSumN / (1 + edge.N)
		return returnValue

	def Pi(self, temperature):
		sumN = 0
		exp = 1 / temperature
		for edge in self.edges:
			sumN += edge.N ** exp
		assert sumN != 0
		returnValue = np.zeros(self.edgeSize, np.float32)
		for edge in self.edges:
			returnValue[edge.index] = edge.N ** exp / sumN
		return returnValue

	def select(self, Cpuct):
		maxActionValue = -1
		maxActionValueEdge = None
		U = self.U(Cpuct)
		for edge in self.edges:
			actionValue = U[edge.index] + edge.Q
			if maxActionValueEdge is None or maxActionValue < actionValue:
				maxActionValue = actionValue
				maxActionValueEdge = edge
		return maxActionValueEdge


class MCTSConfig:

	def __init__(self):
		self.Cpuct = 1.5
		self.temperature = 1
		self.maxNodes = 800


class MCTS:

	def __init__(self, game: Game, config: MCTSConfig):
		self.game = game
		self.nodeCount = 0
		self.rootNode = None
		self.config = config

	def getNodeCount(self):
		return self.nodeCount

	def backup(self, node):
		v = node.v
		edge = node.parentEdge
		while edge:
			v = -v
			edge.backup(v)
			edge = edge.parentNode.parentEdge

	def createNewNode(self, parentEdge = None):
		P, v = self.game.getEvaluation()
		actions = self.game.getActions()
		newNode = MCTNode(actions, P, v, parentEdge)
		self.nodeCount += 1
		return newNode

	def expandNode(self, node):
		edge = node.select(self.config.Cpuct)
		if not edge: # arrive terminate node, need backup too
			self.nodeCount += 1 # virtual node
			return node
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
			self.backup(newNode)
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
		if not self.rootNode:
			return None
		edge = self.rootNode.getEdge(actionIndex)
		if edge is None:
			return None
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
