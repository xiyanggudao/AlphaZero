import AlphaZero.Game
import numpy as np


class Game(AlphaZero.Game.Game):

    def __init__(self, network):
        self.chessBoard = np.zeros([2, 19, 19], dtype=np.int8)
        self.historyActions = []
        self.activeColor = 0 # black 0, white 1
        self.network = network

    def isBoardFull(self):
        for i in range(19):
            for j in range(19):
                if self.chessBoard[0][i][j] == 0 and self.chessBoard[1][i][j] == 0:
                    return False
        return True

    def isPlayerWins(self, playerColor):
        for i in range(19):
            for j in range(19):
                isSerialRow = True
                isSerialCol = True
                isSerialDiagLeft = True
                isSerialDiagRight = True
                for k in range(5):
                    if j+k < 19 and self.chessBoard[playerColor][i][j+k] == 0:
                        isSerialRow = False
                    if i+k < 19 and self.chessBoard[playerColor][i+k][j] == 0:
                        isSerialCol = False
                    if i+k < 19 and j-k >= 0 and self.chessBoard[playerColor][i+k][j-k] == 0:
                        isSerialDiagLeft = False
                    if i+k < 19 and j+k < 19 and  self.chessBoard[playerColor][i+k][j+k] == 0:
                        isSerialDiagRight = False
                if isSerialRow or isSerialCol or isSerialDiagLeft or isSerialDiagRight:
                    return True
        return False

    # get (P, v) of current game state
    def getEvaluation(self):
        return self.network.run(self.getInputPlanes(), self.getInputPolicyMask())

    # get actions of current game state, can be list or dict with index key
    def getActions(self):
        actions = {}
        for i in range(19):
            for j in range(19):
                if self.chessBoard[0][i][j] == 0 and self.chessBoard[1][i][j] == 0:
                    actions[i*19+j] = (i, j)
        return actions

    def takeAction(self, action):
        self.chessBoard[self.activeColor][action[0]][action[1]] = 1
        self.activeColor ^= 1
        self.historyActions.append(action)

    def undoAction(self):
        if not len(self.historyActions) > 0:
            return
        action = self.historyActions[-1]
        del self.historyActions[-1]
        self.activeColor ^= 1
        self.chessBoard[self.activeColor][action[0]][action[1]] = 0

    def reset(self):
        self.chessBoard.fill(0)
        self.historyActions = []
        self.activeColor = 0

    # judge whether current game state is game over
    def isTerminated(self):
        # the board is full of chessmen
        if self.isBoardFull():
            return True
        return self.isPlayerWins(0) or self.isPlayerWins(1)

    # get game result, for example: current player wins 1, loss -1, draws 0
    def getTerminateValue(self):
        if self.isPlayerWins(self.activeColor):
            return 1
        elif self.isPlayerWins(self.activeColor ^ 1):
            return -1
        else:
            return 0

    # get network input planes
    def getInputPlanes(self):
        planes = np.empty([19, 19, 3], dtype=np.float32)
        for i in range(19):
            for j in range(19):
                planes[i][j][0] = self.chessBoard[0][i][j]
                planes[i][j][1] = self.chessBoard[1][i][j]
                planes[i][j][2] = self.activeColor
        return planes

    # get network input planes, to filter legal actions
    def getInputPolicyMask(self):
        mask = np.zeros(19*19, dtype=np.int8)
        if self.isTerminated():
            return mask
        for i in range(19):
            for j in range(19):
                if self.chessBoard[0][i][j] == 0 and self.chessBoard[1][i][j] == 0:
                    mask[i * 19 + j] = 1
        return mask
