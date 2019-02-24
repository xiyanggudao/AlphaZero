import multiprocessing
import sys
from AlphaZero.SelfPlayer import SelfPlayer, NetworkBatch
from AlphaZero.Creator import Creator


class TrainConfig:

    def __init__(self):
        self.trainBatchSize = 256
        self.runBatchSize = 8
        self.maxBatchs = 2**16
        self.processCount = 2


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
        modelFileLock = multiprocessing.Lock()
        batchCount = startBatch

        # self play data child process
        processPool = []
        for i in range(1, self.trainConfig.processCount):
            processPool.append(multiprocessing.Process(target=generateSelfPlayData, args=(self.creator, trainDataQueue, modelFileLock,)))
            processPool[-1].start()

        # run self player
        network = NetworkBatch(self.network, self.trainConfig.runBatchSize)
        selfPlayConfig = self.creator.createSelfPlayConfig()
        selfPlayer = SelfPlayer(network, self.creator, selfPlayConfig, trainDataQueue)
        selfPlayer.start()

        try:
            while batchCount < self.trainConfig.maxBatchs:
                # wait for data
                while trainDataQueue.qsize() < self.trainConfig.trainBatchSize:
                    network.condition.acquire()
                    network.condition.wait()
                    network.condition.release()

                # train
                inputPlanes, inputPolicyMask, predictionProbability, predictionValue = self.getBatchData(trainDataQueue)
                print('train start', batchCount, end='')
                sys.stdout.flush()
                network.train(inputPlanes, inputPolicyMask, predictionProbability, predictionValue, batchCount)
                modelFileLock.acquire()
                network.save()
                modelFileLock.release()
                batchCount += 1
                print('train end', end='')
                sys.stdout.flush()
        finally:
            for process in processPool:
                process.terminate()
            return batchCount

    def runData(self, dataQueue, modelFileLock):
        # run self player
        network = NetworkBatch(self.network, self.trainConfig.runBatchSize)
        selfPlayConfig = self.creator.createSelfPlayConfig()
        selfPlayer = SelfPlayer(network, self.creator, selfPlayConfig, dataQueue)
        selfPlayer.start()

        # update model
        gameCount = 0
        while True:
            network.condition.acquire()
            network.condition.wait()
            network.condition.release()

            if gameCount != selfPlayer.gameCount and selfPlayer.gameCount % network.batchSize == 0:
                gameCount = selfPlayer.gameCount
                print('load', end='')
                sys.stdout.flush()
                modelFileLock.acquire()
                network.load()
                modelFileLock.release()

def generateSelfPlayData(creator, queue, modelFileLock):
    trainer = creator.createTrainer()
    trainer.runData(queue, modelFileLock)
