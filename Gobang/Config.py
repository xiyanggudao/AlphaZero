from AlphaZero.Network import NetworkConfig
from AlphaZero.MCTS import MCTSConfig
from AlphaZero.Trainer import TrainConfig
from AlphaZero.CreatorBase import CreatorBase
from AlphaZero.SelfPlayer import SelfPlayConfig
from Gobang.Game import Game
import os


class GobangCreator(CreatorBase):

    def createGame(self, network) -> Game:
        game = Game(network)
        return game

    def createNetworkConfig(self) -> NetworkConfig:
        config = NetworkConfig()
        config.setInputPlane(3, 11, 11)
        config.setOutputProbabilitySize(11 * 11)
        path = os.path.dirname(__file__)
        config.setModelFilePath(path + '/model/model.ckpt')
        config.blockConvolutionFilters = 64
        config.residualBlocks = 1
        config.learningRate = 0.01
        return config

    def createMCTSConfig(self) -> MCTSConfig:
        config = MCTSConfig()
        return config

    def createTrainConfig(self) -> TrainConfig:
        config = TrainConfig()
        config.trainBatchSize = 128
        return config

    def createSelfPlayConfig(self) -> SelfPlayConfig:
        config = SelfPlayConfig()
        return config
