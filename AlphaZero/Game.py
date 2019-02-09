
class Game:

    # get (P, v) of current game state
    def getEvaluation(self):
        pass

    # get actions of current game state, can be list or dict with index key
    def getActions(self):
        pass

    def takeAction(self, action):
        pass

    def undoAction(self):
        pass

    def reset(self):
        pass

    # judge whether current game state is game over
    def isTerminated(self):
        pass

    # get game result, for example: current player wins 1, loss -1, draws 0
    def getTerminateValue(self):
        pass

    # get network input planes
    def getInputPlanes(self):
        pass

    # get network input planes, to filter legal actions
    def getInputPolicyMask(self):
        pass
