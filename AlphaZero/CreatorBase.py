from AlphaZero.Network import Network, NetworkConfig
from AlphaZero.MCTS import MCTS, MCTSConfig
from AlphaZero.Game import Game
from AlphaZero.Trainer import Trainer, TrainConfig
from AlphaZero.SelfPlayer import SelfPlayConfig
from AlphaZero.Creator import Creator


class CreatorBase(Creator):

    def createNetwork(self) -> Network:
        config = self.createNetworkConfig()
        network = Network(config)
        network.buildNetwork()
        return network

    def createMCTS(self, network) -> MCTS:
        config = self.createMCTSConfig()
        game = self.createGame(network)
        mcts = MCTS(game, config)
        return mcts

    def createTrainer(self) -> Trainer:
        config = self.createTrainConfig()
        trainer = Trainer(config, self)
        return trainer

    def createGame(self, network) -> Game:
        pass

    def createNetworkConfig(self) -> NetworkConfig:
        pass

    def createMCTSConfig(self) -> MCTSConfig:
        pass

    def createTrainConfig(self) -> TrainConfig:
        pass

    def createSelfPlayConfig(self) -> SelfPlayConfig:
        pass
