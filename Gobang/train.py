import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Gobang.Config import GobangCreator


if __name__ == '__main__':
    creator = GobangCreator()
    startBatchFile = os.path.join(os.path.dirname(__file__), 'steps')

    trainer = creator.createTrainer()
    endBatchs = trainer.runTrain(startBatchFile)
