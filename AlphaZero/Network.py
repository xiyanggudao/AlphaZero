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
		self.cOfL2Loss = 1E-3
		self.cOfPolicyLoss = 1

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
		self.batchNormGamma = 0.1

		# train parameters
		self.learningRate = 0.001

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
		self.isTraining = None

	def batchNormalization(self, x, isTraining):
		return tf.layers.batch_normalization(x, training=isTraining,
			gamma_initializer=tf.initializers.constant(self.config.batchNormGamma))

	def addConvolutionLayer(self, input, channels, kSize, isTraining):
		z = tf.layers.conv2d(input, channels, (kSize, kSize), padding="same", use_bias=False,
				kernel_initializer=tf.initializers.random_normal())
		normal = self.batchNormalization(z, isTraining)
		y = tf.nn.relu(normal)
		return y

	def addResidualConvolutionBlock(self, input, channels, kSize, isTraining):
		layer1 = self.addConvolutionLayer(input, channels, kSize, isTraining)
		z = tf.layers.conv2d(layer1, channels, (kSize, kSize), padding="same", use_bias=False,
			kernel_initializer=tf.initializers.random_normal())
		normal = self.batchNormalization(z, isTraining)
		y = tf.nn.relu(normal + input)
		return y

	def addSoftmaxLayer(self, input, mask):
		expVal = tf.math.exp(input) * mask
		y = expVal / tf.math.reduce_sum(expVal, 1, keepdims=True)
		return y

	def createNetworkGraph(self):
		graph = tf.Graph()
		with graph.as_default():
			# input layer
			inputLayer = tf.placeholder(tf.float32, [None
				, self.config.inputPlaneColumns
				, self.config.inputPlaneRows
				, self.config.inputPlaneLayers])
			inputPolicyMaskLayer = tf.placeholder(tf.float32, [None, self.config.outputProbabilitySize])
			isTraining = tf.placeholder(tf.bool)

			# middle hidden layer
			middleLayer = self.addConvolutionLayer(inputLayer
				, self.config.blockConvolutionFilters
				, self.config.blockKernelSize, isTraining)
			tf.summary.histogram("res/first", middleLayer)
			for i in range(self.config.residualBlocks):
				middleLayer= self.addResidualConvolutionBlock(middleLayer
					, self.config.blockConvolutionFilters
					, self.config.blockKernelSize, isTraining)
			tf.summary.histogram("res/middle", middleLayer)

			# policy layer
			policyLayer = self.addConvolutionLayer(middleLayer
				, self.config.policyConvolutionFilters
				, self.config.policyKernelSize, isTraining)
			policyLayer = tf.layers.flatten(policyLayer)
			policyLayer = tf.layers.dense(policyLayer, self.config.outputProbabilitySize,
				kernel_initializer=tf.initializers.truncated_normal(0, 1/policyLayer.shape.as_list()[1]))
			tf.summary.histogram("res/policy", policyLayer)
			policyLayer = self.addSoftmaxLayer(policyLayer, inputPolicyMaskLayer)

			# value layer
			valueLayer = self.addConvolutionLayer(middleLayer
				, self.config.valueConvolutionFilters
				, self.config.valueKernelSize, isTraining)
			valueLayer = tf.layers.flatten(valueLayer)
			for i in range(self.config.valueHiddenLayers):
				valueLayer = tf.layers.dense(valueLayer, self.config.valueHiddenLayerSize,
					activation=tf.nn.relu,
					kernel_initializer=tf.initializers.truncated_normal(0, 1/valueLayer.shape.as_list()[1]))
			valueLayer = tf.layers.dense(valueLayer, 1, activation=tf.nn.tanh,
				kernel_initializer=tf.initializers.truncated_normal(0, 1/valueLayer.shape.as_list()[1]))
			valueLayer = tf.layers.flatten(valueLayer)
			tf.summary.histogram("res/value", valueLayer)

			# loss
			predictionProbability = tf.placeholder(tf.float32, [None, self.config.outputProbabilitySize])
			predictionValue = tf.placeholder(tf.float32, [None])
			lossL2 = tf.math.reduce_mean([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
			lossValue = tf.math.reduce_mean(tf.math.square(predictionValue - valueLayer))
			lossPolicy = predictionProbability * tf.math.log(tf.clip_by_value(policyLayer, 1E-8, 1))
			lossPolicy = - tf.math.reduce_mean(tf.math.reduce_sum(lossPolicy, 1))
			lossPolicy = self.config.cOfPolicyLoss * lossPolicy
			lossL2 = self.config.cOfL2Loss * lossL2
			loss = lossValue + lossPolicy + lossL2

			# train
			updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(updateOps):
				train = tf.train.GradientDescentOptimizer(self.config.learningRate).minimize(loss)

			# summary
			tf.summary.scalar('loss', loss)
			tf.summary.scalar('lossValue', lossValue)
			tf.summary.scalar('lossPolicy', lossPolicy)
			tf.summary.scalar('lossL2', lossL2)
			self.summaryOp = tf.summary.merge_all()
			self.summary_writer = tf.summary.FileWriter('logs')

			session = tf.Session(graph=graph)
			session.run(tf.global_variables_initializer())
		self.session = session
		self.graph = graph
		self.inputPlanes = inputLayer
		self.inputPolicyMask = inputPolicyMaskLayer
		self.outputProbability = policyLayer
		self.outputValue = valueLayer
		self.predictionProbability = predictionProbability
		self.predictionValue = predictionValue
		self.trainFunction = train
		self.isTraining = isTraining

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
		result = self.runBatch([inputPlanes], [inputPolicyMask])
		P = result[0][0]
		v = result[1][0]
		sumP = np.sum(P)
		assert np.abs(sumP-1) < 1E-4
		return P, v

	def runBatch(self, inputPlanes, inputPolicyMask):
		feed = {
			self.inputPlanes: inputPlanes,
			self.inputPolicyMask: inputPolicyMask,
			self.isTraining: False,
		}
		result = self.session.run({'P': self.outputProbability, 'v': self.outputValue}, feed_dict = feed)
		P = result['P']
		v = result['v']
		return P, v

	def train(self, inputPlanes, inputPolicyMask, predictionProbability, predictionValue, trainCount):
		feed = {
			self.inputPlanes: inputPlanes,
			self.inputPolicyMask: inputPolicyMask,
			self.predictionProbability: predictionProbability,
			self.predictionValue: predictionValue,
			self.isTraining: True,
		}
		result = self.session.run({'train':self.trainFunction, 'summary':self.summaryOp}, feed_dict = feed)
		self.summary_writer.add_summary(result['summary'], trainCount)

