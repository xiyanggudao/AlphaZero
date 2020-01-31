from AlphaZero.Network import Network
from AlphaZero.Creator import Creator
from AlphaZero.TrainData import TrainData
from AlphaZero.MCTS import MCTS
import threading
import numpy as np
import sys
import multiprocessing


class SelfPlayConfig:

    def __init__(self):
        self.clientCount = 16
        self.noiseScale = 0.2
        self.dirichletAlpha = 0.2


class NetworkInput:

    def __init__(self):
        self.inputPlanes = None
        self.inputPolicyMask = None


class NetworkOutput:

    def __init__(self):
        self.outputProbability = None
        self.outputValue = None


class NetworkClient:

    def __init__(self, network: Network, queueInput, queueOutput):
        self.queueInput = queueInput
        self.queueOutput = queueOutput
        self.config = network.config

    def run(self, inputPlanes, inputPolicyMask):
        assert np.sum(inputPolicyMask) != 0

        input = NetworkInput()
        input.inputPlanes = inputPlanes
        input.inputPolicyMask = inputPolicyMask
        self.queueInput.put(input)
        result = self.queueOutput.get()

        sumP = np.sum(result.outputProbability)
        assert np.abs(sumP-1) < 1E-4

        return result.outputProbability, result.outputValue


class NetworkServer:

    def __init__(self, network: Network, batchSize: int):
        self.network = network
        self.lock = threading.Lock()
        self.batchSize = batchSize
        self.inputQueues = []
        self.outputQueues = []

    def createNetworkClient(self) -> NetworkClient:
        inQueue = multiprocessing.Queue()
        ouQueue = multiprocessing.Queue()
        client = NetworkClient(self.network, inQueue, ouQueue)
        self.inputQueues.append(inQueue)
        self.outputQueues.append(ouQueue)
        return client

    def runBatch(self, inputs):
        assert len(inputs) == self.batchSize
        inputPlanes = []
        inputPolicyMask = []
        for data in inputs:
            inputPlanes.append(data.inputPlanes)
            inputPolicyMask.append(data.inputPolicyMask)
        self.lock.acquire()
        P, v = self.network.runBatch(inputPlanes, inputPolicyMask)
        self.lock.release()
        outputs = []
        for i in range(self.batchSize):
            outData = NetworkOutput()
            outData.outputProbability = P[i]
            outData.outputValue = v[i]
            outputs.append(outData)
        return outputs

    def run(self):
        qCount = len(self.inputQueues)
        assert qCount == len(self.outputQueues)
        assert self.batchSize < qCount

        startIndex = 0
        endIndex = 0
        inputs = []
        while True:
            inputs.clear()
            while len(inputs) < self.batchSize:
                inputs.append(self.inputQueues[endIndex].get())
                endIndex = (endIndex + 1) % qCount
            outputs = self.runBatch(inputs)
            assert len(inputs) == len(outputs)
            for data in outputs:
                self.outputQueues[startIndex].put(data)
                startIndex = (startIndex + 1) % qCount

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


class SelfPlayerClient:

    def __init__(self, network: NetworkClient, creator: Creator, dataQueue):
        self.network = network
        self.creator = creator
        self.dataQueue = dataQueue
        self.process = None
        self.config = None

    def start(self):
        assert self.process is None
        process = multiprocessing.Process(target=self.run)
        process.start()
        self.process = process

    def run(self):
        mcts = self.creator.createMCTS(self.network)
        self.config = self.creator.createSelfPlayConfig()
        while True:
            self.selfPlay(mcts, self.dataQueue)

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


class SelfPlayerServer:

    def __init__(self, network: NetworkServer, creator: Creator, dataQueue):
        self.network = network
        self.creator = creator
        self.dataQueue = dataQueue
        self.clients = []

    def start(self):
        assert self.network.batchSize <= len(self.clients)

        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        for client in self.clients:
            client.start()

    def createClients(self):
        config = self.creator.createSelfPlayConfig()
        for i in range(config.clientCount):
            self.createSelfPlayerClient()

    def createSelfPlayerClient(self) -> SelfPlayerClient:
        networkClient = self.network.createNetworkClient()
        client = SelfPlayerClient(networkClient, self.creator, self.dataQueue)
        self.clients.append(client)
        return client

    def terminateClients(self):
        for client in self.clients:
            client.process.terminate()

    def run(self):
        self.network.run()
