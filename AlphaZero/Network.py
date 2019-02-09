import tensorflow as tf
import numpy as np
import os


class NetworkConfig:

	def __init__(self):
		# model saved path
		self.modelFilePath = None

		# input plane features
		self.inputPlaneLayers = 0
		self.inputPlaneRows = 0
		self.inputPlaneColumns = 0

		# output features
		self.outputProbabilitySize = 0

		# a parameter controlling the level of L2 weight regularisation in loss function
		self.cOfLoss = 1E-4

		# network parameters
		self.residualBlocks = 8
		self.blockConvolutionFilters = 128
		self.blockKernelSize = 3
		self.policyConvolutionFilters = 1
		self.policyKernelSize = 1
		self.valueConvolutionFilters = 1
		self.valueKernelSize = 1
		self.valueHiddenLayers = 1
		self.valueHiddenLayerSize = 64

		# train parameters
		self.learningRate = 1E-4

	def setInputPlane(self, planeLayers, planeRows, planeColumns):
		self.inputPlaneLayers = planeLayers
		self.inputPlaneRows = planeRows
		self.inputPlaneColumns = planeColumns

	def setOutputProbabilitySize(self, outputProbabilitySize):
		self.outputProbabilitySize = outputProbabilitySize

	def setModelFilePath(self, filePath):
		self.modelFilePath = filePath


class Network:

	def __init__(self, config):
		self.config = config
		self.session = None
		self.graph = None
		self.inputPlanes = None
		self.inputPolicyMask = None
		self.outputProbability = None
		self.outputValue = None
		self.predictionProbability = None
		self.predictionValue = None
		self.trainFunction = None

	def batchNormalization(self, x):
		mean, variance = tf.nn.moments(x, [0])
		return tf.nn.batch_normalization(x, mean, variance, 0, 1, 1e-8)

	def addConvolutionLayer(self, input, in_channels, out_channels, kSize):
		weights = tf.Variable(tf.random.normal([kSize, kSize, in_channels, out_channels]))
		bias = tf.Variable(tf.random.normal([out_channels]))
		z = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
		normal = self.batchNormalization(z) + bias
		y = tf.nn.relu(normal)
		return y

	def addResidualConvolutionBlock(self, input, channels, kSize):
		layer1 = self.addConvolutionLayer(input, channels, channels, kSize)
		weights = tf.Variable(tf.random.normal([kSize, kSize, channels, channels]))
		bias = tf.Variable(tf.random.normal([channels]))
		z = tf.nn.conv2d(layer1, weights, strides=[1, 1, 1, 1], padding='SAME')
		normal = self.batchNormalization(z) + bias
		y = tf.nn.relu(normal + input)
		return y

	def addPureDenseLayer(self, input, inputSize, outputSize):
		weights = tf.Variable(tf.random.normal([inputSize, outputSize]))
		bias = tf.Variable(tf.random.normal([1, outputSize]))
		z = tf.matmul(input, weights)
		normal = self.batchNormalization(z) + bias
		y = normal
		return y

	def addDenseLayer(self, input, inputSize, outputSize):
		z = self.addPureDenseLayer(input, inputSize, outputSize)
		y = tf.nn.relu(z)
		return y

	def addSoftmaxLayer(self, input, mask):
		# softmax[i] = exp(input[i]) / sum(exp(input))
		# pickySoftmax[i] = pickySwitch[i]*exp(input[i]) / sum(pickySwitch*exp(input))
		expVal = tf.math.exp(input) * mask
		y = expVal / tf.math.reduce_sum(expVal, 1, keepdims=True)
		return y

	def AddTanhDenceLayer(self, input, inputSize, outputSize):
		weights = tf.Variable(tf.random.normal([inputSize, outputSize]))
		bias = tf.Variable(tf.random.normal([1, outputSize]))
		z = tf.matmul(input, weights) + bias
		y = tf.nn.tanh(z)
		return y

	def createNetworkGraph(self):
		assert self.config
		graph = tf.Graph()
		with graph.as_default():
			# input layer
			inputLayer = tf.placeholder(tf.float32, [None
				, self.config.inputPlaneColumns
				, self.config.inputPlaneRows
				, self.config.inputPlaneLayers])
			inputPolicyMaskLayer = tf.placeholder(tf.float32, [None, self.config.outputProbabilitySize])
			# middle hidden layer
			middleLayer = self.addConvolutionLayer(inputLayer
				, self.config.inputPlaneLayers
				, self.config.blockConvolutionFilters
				, self.config.blockKernelSize)
			for i in range(self.config.residualBlocks):
				middleLayer= self.addResidualConvolutionBlock(middleLayer
					, self.config.blockConvolutionFilters
					, self.config.blockKernelSize)
			# policy layer
			policyLayer = self.addConvolutionLayer(middleLayer
				, self.config.blockConvolutionFilters
				, self.config.policyConvolutionFilters
				, self.config.policyKernelSize)
			policyFlattenSize = self.config.inputPlaneColumns * self.config.inputPlaneRows * self.config.policyConvolutionFilters
			policyLayer = tf.reshape(policyLayer, [-1, policyFlattenSize]) # flatten
			policyLayer = self.addPureDenseLayer(policyLayer, policyFlattenSize, self.config.outputProbabilitySize)
			policyLayer = self.addSoftmaxLayer(policyLayer, inputPolicyMaskLayer)
			# value layer
			valueLayer = self.addConvolutionLayer(middleLayer
				, self.config.blockConvolutionFilters
				, self.config.valueConvolutionFilters
				, self.config.valueKernelSize)
			valueHiddenLayerSize = self.config.inputPlaneColumns * self.config.inputPlaneRows * self.config.valueConvolutionFilters
			valueLayer = tf.reshape(valueLayer, [-1, valueHiddenLayerSize])  # flatten
			for i in range(self.config.valueHiddenLayers):
				valueLayer = self.addDenseLayer(valueLayer, valueHiddenLayerSize, self.config.valueHiddenLayerSize)
				valueHiddenLayerSize = self.config.valueHiddenLayerSize
			valueLayer = self.AddTanhDenceLayer(valueLayer, valueHiddenLayerSize, 1)
			valueLayer = tf.reshape(valueLayer, [-1])

			# train variant
			predictionProbability = tf.placeholder(tf.float32, [None, self.config.outputProbabilitySize])
			predictionValue = tf.placeholder(tf.float32, [None])
			lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
			loss = tf.math.reduce_mean(tf.math.square(predictionValue - valueLayer))\
				- tf.math.reduce_mean(predictionProbability * tf.math.log(policyLayer))\
				+ self.config.cOfLoss * tf.math.reduce_mean(lossL2)
			train = tf.train.GradientDescentOptimizer(self.config.learningRate).minimize(loss)

			session = tf.Session(graph=graph)
		self.session = session
		self.graph = graph
		self.inputPlanes = inputLayer
		self.inputPolicyMask = inputPolicyMaskLayer
		self.outputProbability = policyLayer
		self.outputValue = valueLayer
		self.predictionProbability = predictionProbability
		self.predictionValue = predictionValue
		self.trainFunction = train

	def buildNetwork(self):
		assert self.config
		self.createNetworkGraph()
		if self.config.modelFilePath and os.path.exists(os.path.dirname(self.config.modelFilePath)):
			self.load()
		else:
			with self.graph.as_default():
				self.session.run(tf.global_variables_initializer())
			if self.config.modelFilePath:
				self.save()

	def save(self):
		assert self.config and self.config.modelFilePath
		with self.graph.as_default():
			saver = tf.train.Saver()
			saver.save(self.session, self.config.modelFilePath)

	def load(self):
		assert self.config and self.config.modelFilePath
		with self.graph.as_default():
			saver = tf.train.Saver()
			saver.restore(self.session, self.config.modelFilePath)

	def run(self, inputPlanes, inputPolicyMask):
		assert np.sum(inputPolicyMask) != 0
		feed = {
			self.inputPlanes: [inputPlanes],
			self.inputPolicyMask: [inputPolicyMask],
		}
		result = self.session.run({'P': self.outputProbability, 'v': self.outputValue}, feed_dict = feed)
		P = result['P'][0]
		v = result['v'][0]
		sumP = np.sum(P)
		assert np.abs(sumP-1) < 1E-6
		return P, v

	def train(self, inputPlanes, inputPolicyMask, predictionProbability, predictionValue):
		feed = {
			self.inputPlanes: inputPlanes,
			self.inputPolicyMask: inputPolicyMask,
			self.predictionProbability: predictionProbability,
			self.predictionValue: predictionValue,
		}
		self.session.run(self.trainFunction, feed_dict = feed)

