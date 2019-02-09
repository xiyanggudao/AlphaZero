from AlphaZero.Network import NetworkConfig
from AlphaZero.MCTS import MCTSConfig
from AlphaZero.Trainer import TrainConfig
import os

def createNetworkConfig():
    config = NetworkConfig()
    config.setInputPlane(3, 19, 19)
    config.setOutputProbabilitySize(19*19)
    path = os.path.dirname(__file__)
    config.setModelFilePath(path + '/model/model.ckpt')
    config.blockConvolutionFilters = 64
    config.residualBlocks = 6
    return config

def createMCTSConfig():
    config = MCTSConfig()
    config.maxNodes = 2**10
    return config

def createTrainConfig():
    config = TrainConfig()
    return config
