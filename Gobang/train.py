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

trainer.run()
