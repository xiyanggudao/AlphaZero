import multiprocessing
import numpy as np
import sys
import os


class TrainConfig:

    def __init__(self):
        self.batchSize = 128
        self.maxBatchs = 2**16
        self.modelSaveBatchs = 8


class TrainData:

    def __init__(self):
        self.inputPlanes = None
        self.inputPolicyMask = None
        self.predictionProbability = None
        self.predictionValue = None


class Trainer:

    def __init__(self, network, MCTS, trainConfig):
        self.network = network
        self.MCTS = MCTS
        self.trainConfig = trainConfig

    def selectActionIndex(self, Pi):
        try:
            return np.random.choice(len(Pi), p=Pi)
        except:
            print(Pi)

    def selfPlay(self, queue):
        dataOneGame = []
        self.MCTS.reset()
        while not self.MCTS.game.isTerminated():
            self.MCTS.expandMaxNodes()
            Pi = self.MCTS.Pi()
            if not len(Pi) > 0:
                break
            stepData = TrainData()
            stepData.inputPlanes = self.MCTS.game.getInputPlanes()
            stepData.inputPolicyMask = self.MCTS.game.getInputPolicyMask()
            stepData.predictionProbability = Pi
            actionIndex = self.selectActionIndex(Pi)
            action = self.MCTS.play(actionIndex)
            assert action
            dataOneGame.append(stepData)
        resultValue = self.MCTS.game.getTerminateValue()
        dataOneGame.reverse()
        for data in dataOneGame:
            data.predictionValue = resultValue
            resultValue = -resultValue
        for data in dataOneGame:
            queue.put(data)

    def getBatchData(self, queue):
        inputPlanes = []
        inputPolicyMask = []
        predictionProbability = []
        predictionValue = []
        for i in range(self.trainConfig.batchSize):
            data = queue.get()
            inputPlanes.append(data.inputPlanes)
            inputPolicyMask.append(data.inputPolicyMask)
            predictionProbability.append(data.predictionProbability)
            predictionValue.append(data.predictionValue)
        return inputPlanes, inputPolicyMask, predictionProbability, predictionValue

    def run(self, selfPlayDataGenerate):
        processCount = os.cpu_count()
        trainDataQueue = multiprocessing.Queue(self.trainConfig.batchSize*processCount)
        batchCount = 0
        # generate data by self play
        processPool = []
        for i in range(processCount):
            processPool.append(multiprocessing.Process(target=selfPlayDataGenerate, args=(trainDataQueue,)))
            processPool[-1].start()
        try:
            while batchCount < self.trainConfig.maxBatchs:
                # train
                inputPlanes, inputPolicyMask, predictionProbability, predictionValue = self.getBatchData(trainDataQueue)
                print('train start', end='')
                sys.stdout.flush()
                batchCount += 1
                self.network.train(inputPlanes, inputPolicyMask, predictionProbability, predictionValue)
                if batchCount % self.trainConfig.modelSaveBatchs == 0:
                    self.network.save()
                    print('save', end='')
                    sys.stdout.flush()
                print('train end', end='')
                sys.stdout.flush()
        finally:
            for i in range(processCount):
                processPool[i].terminate()

