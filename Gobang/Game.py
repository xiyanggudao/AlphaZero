import AlphaZero.Game
import numpy as np
from AlphaZero.Network import Network


class Game(AlphaZero.Game.Game):

    def __init__(self, network: Network):
        super().__init__(network)
        self.chessBoard = np.empty([2, 19, 19, 3], dtype=np.int8)
        self.policyMask = np.empty(19*19, dtype=np.int8)
        self.network = network
        self.actions = None
        self.reset()

    def isBoardFull(self):
        return len(self.historyActions) >= 19*19

    def isSerialFive(self, i, j):
        if self.chessBoard[0][i][j][0] != 0:
            playerColor = 0
        elif self.chessBoard[0][i][j][1] != 0:
            playerColor = 1
        else:
            return False

        left = 0
        while j - left - 1 >= 0 and self.chessBoard[0][i][j - left - 1][playerColor] != 0:
            left += 1
        if left >= 4:
            return True

        right = 0
        while j + right + 1 < 19 and self.chessBoard[0][i][j + right + 1][playerColor] != 0:
            right += 1
        if left + right >= 4:
            return True

        up = 0
        while i - up - 1 >= 0 and self.chessBoard[0][i - up - 1][j][playerColor] != 0:
            up += 1
        if up >= 4:
            return True

        down = 0
        while i + down + 1 < 19 and self.chessBoard[0][i + down + 1][j][playerColor] != 0:
            down += 1
        if up + down >= 4:
            return True

        leftUp = 0
        while j - leftUp - 1 >= 0 and i - leftUp - 1 >= 0 and self.chessBoard[0][i - leftUp - 1][j - leftUp - 1][playerColor] != 0:
            leftUp += 1
        if leftUp >= 4:
            return True

        rightDown = 0
        while j + rightDown + 1 < 19 and i + rightDown + 1 < 19 and self.chessBoard[0][i + rightDown + 1][j + rightDown + 1][playerColor] != 0:
            rightDown += 1
        if leftUp + rightDown >= 4:
            return True

        leftDown = 0
        while j - leftDown - 1 >= 0 and i + leftDown + 1 < 19 and self.chessBoard[0][i + leftDown + 1][j - leftDown - 1][playerColor] != 0:
            leftDown += 1
        if leftDown >= 4:
            return True

        rightUp = 0
        while j + rightUp + 1 < 19 and i - rightUp - 1 >= 0 and self.chessBoard[0][i - rightUp - 1][j + rightUp + 1][playerColor] != 0:
            rightUp += 1
        if leftDown + rightUp >= 4:
            return True

    def checkBoard(self):
        for i in range(19):
            for j in range(19):
                assert self.chessBoard[0][i][j][2] == 0
                assert self.chessBoard[1][i][j][2] == 1
                assert self.chessBoard[0][i][j][0] == self.chessBoard[1][i][j][0]
                assert self.chessBoard[0][i][j][1] == self.chessBoard[1][i][j][1]

    # get (P, v) of current game state
    def getEvaluation(self):
        #self.checkBoard()
        mask = self.policyMask
        if self.isTerminated():
            return np.zeros(mask.shape), self.getTerminateValue()
        return self.network.run(self.chessBoard[self.activeColor], mask)

    # get actions of current game state, can be list or dict with index key
    def getActions(self):
        if self.actions:
            return self.actions
        actions = []
        for i in range(19):
            for j in range(19):
                actions.append((i, j))
        self.actions = actions
        return actions

    def takeAction(self, action):
        self.chessBoard[0][action[0]][action[1]][self.activeColor] = 1
        self.chessBoard[1][action[0]][action[1]][self.activeColor] = 1
        self.policyMask[action[0]*19+action[1]] = 0
        if self.isSerialFive(action[0], action[1]):
            self.winner = self.activeColor
        self.activeColor ^= 1
        self.historyActions.append(action)

    def undoAction(self):
        if not len(self.historyActions) > 0:
            return
        action = self.historyActions[-1]
        del self.historyActions[-1]
        self.activeColor ^= 1
        self.chessBoard[0][action[0]][action[1]][self.activeColor] = 0
        self.chessBoard[1][action[0]][action[1]][self.activeColor] = 0
        self.policyMask[action[0]*19+action[1]] = 1
        self.winner = None

    def reset(self):
        self.chessBoard.fill(0)
        for i in range(19):
            for j in range(19):
                self.chessBoard[1][i][j][2] = 1
        self.policyMask.fill(1)
        self.historyActions = []
        self.activeColor = 0 # black 0, white 1
        self.winner = None

    # judge whether current game state is game over
    def isTerminated(self):
        # the board is full of chessmen
        if self.isBoardFull():
            return True
        return self.winner is not None

    # get game result, for example: current player wins 1, loss -1, draws 0
    def getTerminateValue(self):
        if self.winner == self.activeColor:
            return 1
        elif self.winner == (self.activeColor ^ 1):
            return -1
        else:
            return 0

    # get network input planes
    def getInputPlanes(self):
        return self.chessBoard[self.activeColor].copy()

    # get network input planes, to filter legal actions
    def getInputPolicyMask(self):
        if self.isTerminated():
            return np.zeros(19*19, dtype=np.int8)
        return self.policyMask.copy()
