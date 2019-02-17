import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from AlphaZero.Network import Network
from AlphaZero.MCTS import MCTS
from AlphaZero.Trainer import Trainer
from Gobang.Game import Game
import Gobang.Config as config


network = Network(config.createNetworkConfig())
network.buildNetwork()
game = Game(network)
mcts = MCTS(game, config.createMCTSConfig())
trainer = Trainer(network, mcts, config.createTrainConfig())

def selfPlayDataGenerate(queue, modelLock):
    while True:
        modelLock.acquire()
        network.load()
        modelLock.release()
        trainer.selfPlay(queue)

if __name__ == '__main__':
    startBatchs = 0
    startBatchFile = os.path.join(os.path.dirname(__file__), 'steps')
    if os.path.exists(startBatchFile):
        with open(startBatchFile, "rt") as batchFile:
            text = batchFile.read()
            startBatchs = int(text)
    endBatchs = trainer.run(selfPlayDataGenerate, startBatchs)
    if endBatchs and endBatchs > startBatchs:
        with open(startBatchFile, "wt") as batchFile:
            batchFile.write(str(endBatchs+1))
