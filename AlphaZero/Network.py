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
		self.cOfL2Loss = 1E-6
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

		# train parameters
		self.learningRate = 1E-2

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

		tf.summary.histogram("params/weight" + str(self.layer_cnt), weights)
		tf.summary.histogram("params/bias" + str(self.layer_cnt), bias)
		tf.summary.histogram("res/z" + str(self.layer_cnt), z)
		tf.summary.histogram("res/znorm" + str(self.layer_cnt), normal)
		self.layer_cnt += 1

		return y

	def addResidualConvolutionBlock(self, input, channels, kSize):
		layer1 = self.addConvolutionLayer(input, channels, channels, kSize)
		weights = tf.Variable(tf.random.normal([kSize, kSize, channels, channels]))
		bias = tf.Variable(tf.random.normal([channels]))
		z = tf.nn.conv2d(layer1, weights, strides=[1, 1, 1, 1], padding='SAME')
		normal = self.batchNormalization(z) + bias
		y = tf.nn.relu(normal + input)

		tf.summary.histogram("params/weight" + str(self.layer_cnt), weights)
		tf.summary.histogram("params/bias" + str(self.layer_cnt), bias)
		tf.summary.histogram("res/z" + str(self.layer_cnt), z)
		tf.summary.histogram("res/znorm" + str(self.layer_cnt), normal)
		self.layer_cnt += 1

		return y

	def addPureDenseLayer(self, input, inputSize, outputSize):
		weights = tf.Variable(tf.random.normal([inputSize, outputSize]))
		bias = tf.Variable(tf.random.normal([1, outputSize]))
		z = tf.matmul(input, weights)
		normal = self.batchNormalization(z) + bias
		y = normal

		tf.summary.histogram("params/weight" + str(self.layer_cnt), weights)
		tf.summary.histogram("params/bias" + str(self.layer_cnt), bias)
		tf.summary.histogram("res/z" + str(self.layer_cnt), z)
		tf.summary.histogram("res/znorm" + str(self.layer_cnt), normal)
		self.layer_cnt += 1

		return y

	def addDenseLayer(self, input, inputSize, outputSize):
		z = self.addPureDenseLayer(input, inputSize, outputSize)
		y = tf.nn.relu(z)
		return y

	def addSoftmaxLayer(self, input, mask):
		# softmax[i] = exp(input[i]) / sum(exp(input))
		# pickySoftmax[i] = pickySwitch[i]*exp(input[i]) / sum(pickySwitch*exp(input))
		self.debug = input
		expVal = tf.math.exp(input) * mask
		y = expVal / tf.math.reduce_sum(expVal, 1, keepdims=True)

		tf.summary.histogram("res/softmax" + str(self.layer_cnt), y)
		self.layer_cnt += 1

		return y

	def AddTanhDenceLayer(self, input, inputSize, outputSize):
		weights = tf.Variable(tf.random.normal([inputSize, outputSize]))
		bias = tf.Variable(tf.random.normal([1, outputSize]))
		z = tf.matmul(input, weights) + bias
		y = tf.nn.tanh(z)

		tf.summary.histogram("params/weight" + str(self.layer_cnt), weights)
		tf.summary.histogram("params/bias" + str(self.layer_cnt), bias)
		tf.summary.histogram("res/z" + str(self.layer_cnt), z)
		tf.summary.histogram("res/tanh" + str(self.layer_cnt), y)
		self.layer_cnt += 1

		return y

	def createNetworkGraph(self):
		assert self.config
		self.layer_cnt = 0
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
			lossL2 = tf.math.reduce_mean(lossL2)
			self.lossL2 = lossL2
			lossValue = tf.math.reduce_mean(tf.math.square(predictionValue - valueLayer))
			self.lossValue = lossValue
			lossPolicy = predictionProbability * tf.math.log(tf.clip_by_value(policyLayer, 1E-8, 1))
			lossPolicy = - tf.math.reduce_mean(tf.math.reduce_sum(lossPolicy, 1))
			self.lossPolicy = lossPolicy
			self.lossPolicy2 = predictionProbability * tf.math.log(policyLayer)
			lossPolicy = self.config.cOfPolicyLoss * lossPolicy
			lossL2 = self.config.cOfL2Loss * lossL2
			loss = lossValue + lossPolicy + lossL2
			train = tf.train.GradientDescentOptimizer(self.config.learningRate).minimize(loss)

			# summary
			tf.summary.scalar('loss', loss)
			tf.summary.scalar('lossValue', lossValue)
			tf.summary.scalar('lossPolicy', loss)
			tf.summary.scalar('lossL2', lossL2)
			self.summaryOp = tf.summary.merge_all()
			self.summary_writer = tf.summary.FileWriter('logs')
			self.trainCount = 0

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
		if not (np.abs(sumP-1) < 1E-6):
			print(P)
			print(self.session.run(self.debug, feed_dict = feed))
		assert np.abs(sumP-1) < 1E-6
		return P, v

	def train(self, inputPlanes, inputPolicyMask, predictionProbability, predictionValue):
		feed = {
			self.inputPlanes: inputPlanes,
			self.inputPolicyMask: inputPolicyMask,
			self.predictionProbability: predictionProbability,
			self.predictionValue: predictionValue,
		}
		#print('')
		#print(inputPolicyMask)
		#print(predictionProbability)
		#print(inputPolicyMask)
		'''
		r0 = self.session.run({'l2':self.lossL2,'p':self.lossPolicy, 'po':self.outputProbability,'v':self.lossValue}, feed_dict = feed)
		r = self.session.run(
			{'p2': self.lossPolicy2, 'po': self.outputProbability,
			 'd': self.debug}, feed_dict=feed)
		for i in range(5):
			for j in range(19*19):
				if np.isnan(r['p2'][i][j]):
					print(i, j, inputPolicyMask[i][j], predictionProbability[i][j], r['po'][i][j], r['d'][i][j])

		print(r0)
		'''
		result = self.session.run({'train':self.trainFunction, 'summary':self.summaryOp}, feed_dict = feed)
		self.summary_writer.add_summary(result['summary'], self.trainCount)
		self.trainCount += 1

