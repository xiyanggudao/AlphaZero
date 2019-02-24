import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Gobang.Config import GobangCreator


if __name__ == '__main__':
    creator = GobangCreator()
    startBatchs = 0
    startBatchFile = os.path.join(os.path.dirname(__file__), 'steps')
    if os.path.exists(startBatchFile):
        with open(startBatchFile, "rt") as batchFile:
            text = batchFile.read()
            startBatchs = int(text)

    trainer = creator.createTrainer()
    endBatchs = trainer.runTrain(startBatchs)

    if endBatchs and endBatchs > startBatchs:
        with open(startBatchFile, "wt") as batchFile:
            batchFile.write(str(endBatchs+1))
