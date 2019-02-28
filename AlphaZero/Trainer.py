import multiprocessing
import sys
from AlphaZero.SelfPlayer import SelfPlayerServer, NetworkServer
from AlphaZero.Creator import Creator


class TrainConfig:

    def __init__(self):
        self.trainBatchSize = 256
        self.runBatchSize = 8
        self.maxBatchs = 2**16


class Trainer:

    def __init__(self, trainConfig: TrainConfig, creator: Creator):
        self.network = creator.createNetwork()
        self.trainConfig = trainConfig
        self.creator = creator

    def getBatchData(self, queue):
        inputPlanes = []
        inputPolicyMask = []
        predictionProbability = []
        predictionValue = []
        for i in range(self.trainConfig.trainBatchSize):
            data = queue.get()
            inputPlanes.append(data.inputPlanes)
            inputPolicyMask.append(data.inputPolicyMask)
            predictionProbability.append(data.predictionProbability)
            predictionValue.append(data.predictionValue)
        return inputPlanes, inputPolicyMask, predictionProbability, predictionValue

    def runTrain(self, startBatch=0):
        trainDataQueue = multiprocessing.Queue(self.trainConfig.trainBatchSize*2)
        batchCount = startBatch

        # run self player
        network = NetworkServer(self.network, self.trainConfig.runBatchSize)
        selfPlayer = SelfPlayerServer(network, self.creator, trainDataQueue)
        selfPlayer.createClients()
        selfPlayer.start()

        try:
            while batchCount < self.trainConfig.maxBatchs:
                # train
                inputPlanes, inputPolicyMask, predictionProbability, predictionValue = self.getBatchData(trainDataQueue)
                print('train start', batchCount, end='')
                sys.stdout.flush()
                network.train(inputPlanes, inputPolicyMask, predictionProbability, predictionValue, batchCount)
                network.save()
                batchCount += 1
                print('train end', end='')
                sys.stdout.flush()
        finally:
            selfPlayer.terminateClients()
            return batchCount
