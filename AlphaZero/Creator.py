from AlphaZero.Network import Network, NetworkConfig
from AlphaZero.MCTS import MCTS, MCTSConfig
from AlphaZero.Game import Game


class Creator:

    def createNetwork(self) -> Network:
        pass

    def createMCTS(self, network) -> MCTS:
        pass

    def createTrainer(self):
        pass

    def createGame(self, network) -> Game:
        pass

    def createNetworkConfig(self) -> NetworkConfig:
        pass

    def createMCTSConfig(self) -> MCTSConfig:
        pass

    def createTrainConfig(self):
        pass

    def createSelfPlayConfig(self):
        pass
