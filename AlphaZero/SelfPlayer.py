from AlphaZero.Network import Network
from AlphaZero.Creator import Creator
from AlphaZero.TrainData import TrainData
from AlphaZero.MCTS import MCTS
import threading
import numpy as np
import sys




class SelfPlayConfig:

    def __init__(self):
        self.threadCount = 16
        self.noiseScale = 0.2
        self.dirichletAlpha = 0.2


class NetworkBatch:

    def __init__(self, network: Network, batchSize: int):
        self.network = network
        self.batchSize = batchSize
        self.blockThreadIds = []
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.batchDataInputPlanes = {}
        self.batchDataInputPolicyMask = {}
        self.batchDataOutput = {}

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        return attr

    def __runBatchData(self):
        assert len(self.blockThreadIds) == self.batchSize
        inputPlanes = []
        inputPolicyMask = []
        for id in self.blockThreadIds:
            inputPlanes.append(self.batchDataInputPlanes[id])
            inputPolicyMask.append(self.batchDataInputPolicyMask[id])
        P, v = self.network.runBatch(inputPlanes, inputPolicyMask)
        for i in range(len(self.blockThreadIds)):
            self.batchDataOutput[self.blockThreadIds[i]] = (P[i], v[i])

    def __fetchOutput(self, threadId):
        P, v = self.batchDataOutput[threadId]
        self.batchDataOutput[threadId] = None
        return P, v

    def run(self, inputPlanes, inputPolicyMask):
        assert np.sum(inputPolicyMask) != 0

        self.lock.acquire()

        threadId = threading.get_ident()
        self.blockThreadIds.append(threadId)
        self.batchDataInputPlanes[threadId] = inputPlanes
        self.batchDataInputPolicyMask[threadId] = inputPolicyMask
        if len(self.blockThreadIds) < self.batchSize:
            self.condition.wait()
            P, v = self.__fetchOutput(threadId)
        else:
            self.__runBatchData()
            self.blockThreadIds.clear()
            self.condition.notify_all()
            P, v = self.__fetchOutput(threadId)

        self.lock.release()

        sumP = np.sum(P)
        if np.abs(sumP-1) >= 1E-4:
            print('sumP', sumP)
        assert np.abs(sumP-1) < 1E-4
        return P, v

    def load(self):
        self.lock.acquire()
        self.network.load()
        self.lock.release()

    def save(self):
        self.lock.acquire()
        self.network.save()
        self.lock.release()

    def train(self, inputPlanes, inputPolicyMask, predictionProbability, predictionValue, trainCount):
        self.lock.acquire()
        self.network.train(inputPlanes, inputPolicyMask, predictionProbability, predictionValue, trainCount)
        self.lock.release()


class SelfPlayer:

    def __init__(self, network: NetworkBatch, creator: Creator, config: SelfPlayConfig, dataQueue):
        self.network = network
        self.creator = creator
        self.config = config
        self.dataQueue = dataQueue
        self.gameCount = 0

    def start(self):
        assert self.network.batchSize <= self.config.threadCount

        for i in range(self.config.threadCount):
            thread = threading.Thread(target=self.run, daemon=True)
            thread.start()

    def run(self):
        mcts = self.creator.createMCTS(self.network)
        while True:
            self.selfPlay(mcts, self.dataQueue)
            self.gameCount += 1

    def selectActionIndex(self, Pi, mask):
        legalActionCount = 0
        for i in range(len(mask)):
            if mask[i] != 0:
                legalActionCount += 1
        noise = np.random.dirichlet(self.config.dirichletAlpha*np.ones(legalActionCount))
        noiseIndex = 0
        scale = self.config.noiseScale
        for i in range(len(mask)):
            if mask[i] != 0:
                Pi[i] = Pi[i]*(1-scale) + noise[noiseIndex]*scale
                noiseIndex += 1
        try:
            return np.random.choice(len(Pi), p=Pi)
        except:
            print(Pi)

    def selfPlay(self, mcts: MCTS, queue):
        dataOneGame = []
        mcts.reset()
        while not mcts.game.isTerminated():
            mcts.expandMaxNodes()
            Pi = mcts.Pi()
            if not len(Pi) > 0:
                break
            stepData = TrainData()
            stepData.inputPlanes = mcts.game.getInputPlanes()
            stepData.inputPolicyMask = mcts.game.getInputPolicyMask()
            stepData.predictionProbability = Pi
            actionIndex = self.selectActionIndex(Pi, stepData.inputPolicyMask)
            action = mcts.play(actionIndex)
            assert action
            dataOneGame.append(stepData)
        print('_', end='')
        sys.stdout.flush()
        resultValue = mcts.game.getTerminateValue()
        dataOneGame.reverse()
        for data in dataOneGame:
            data.predictionValue = resultValue
            resultValue = -resultValue
        for data in dataOneGame:
            queue.put(data)
