import tkinter
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import Gobang.Config as config
from Gobang.gui.Chessboard import Chessboard

def setWindowSize(window, width, height):
	geometry = '%dx%d' % (width, height)
	window.geometry(geometry)

def refreshBoard():
	global board
	board.setChessmenOnBoard(game.historyActions)
	board.refresh()

def reset():
	global mcts
	mcts.reset()
	refreshBoard()

def selectActionIndex(Pi):
	index = 0
	maxValue = Pi[0]
	for i in range(len(Pi)):
		if maxValue < Pi[i] - 1E-8:
			maxValue = Pi[i]
			index = i
	return index

def action(actionIndex):
	global mcts
	if not mcts.rootNode:
		mcts.expand()
	action = mcts.play(actionIndex)
	assert action
	refreshBoard()

def mctsAction():
	global game
	global mcts
	if game.isTerminated():
		return
	mcts.expandMaxNodes()
	Pi = mcts.Pi()
	if not len(Pi) > 0:
		return
	actionIndex = selectActionIndex(Pi)
	action(actionIndex)

def networkAction():
	global game
	global mcts
	if game.isTerminated():
		return
	if not mcts.rootNode:
		mcts.expand()
	actionIndex = -1
	maxP = -1
	for edge in mcts.rootNode.edges:
		if maxP < edge.P:
			actionIndex = edge.index
			maxP = edge.P
	if actionIndex >= 0:
		action(actionIndex)

def printPv():
	global game
	global mcts
	global board
	if game.isTerminated():
		return
	if not mcts.rootNode:
		mcts.expand()
	board.refresh()
	board.printValue("%0.4f"%mcts.rootNode.v)
	for edge in mcts.rootNode.edges:
		board.printValue("%d"%(edge.P*1000), edge.action)

def printPi():
	global game
	global mcts
	global board
	if game.isTerminated():
		return
	if not mcts.rootNode:
		mcts.expand()
	board.refresh()
	Pi = mcts.Pi()
	for i in range(len(Pi)):
		if Pi[i] != 0:
			board.printValue("%d"%(Pi[i]*1000), mcts.rootNode.getEdge(i).action)

def printN():
	global game
	global mcts
	global board
	if game.isTerminated():
		return
	if not mcts.rootNode:
		mcts.expand()
	board.refresh()
	for edge in mcts.rootNode.edges:
		board.printValue("%d"%edge.N, edge.action)


def onKey(event):
	global mcts
	code = event.keycode
	if code == 27: # Esc
		rootWindow.destroy()
	elif code == 37: # Left
		pass
	elif code == 39: # Right
		pass
	elif code == 38: # Up
		pass
	elif code == 40: # Down
		pass
	elif code == 13: # Enter
		mctsAction()
	elif code == 82: # r
		reset()
	elif code == 80: # p
		printPv()
	elif code == 68: # d
		printPi()
	elif code == 78: # n
		printN()
	elif code == 69: # e
		mcts.expand()
	elif code == 77: # m
		mcts.expandMaxNodes()
	elif code == 65: # a
		networkAction()

def onClick(pos):
	if pos == None:
		return
	global game
	if game.chessBoard[0][pos[0]][pos[1]][0] == 0\
		and game.chessBoard[0][pos[0]][pos[1]][1] == 0\
		and not game.isTerminated():
		action(pos[0]*19+pos[1])



creator = config.GobangCreator()
network = creator.createNetwork()
mcts = creator.createMCTS(network)
game = mcts.game

rootWindow = tkinter.Tk()
cv = tkinter.Canvas(rootWindow)
board = Chessboard(cv)
board.setMoveEventListener(onClick)
rootWindow.bind('<Key>', onKey)
miniW, miniH = board.minimumSize()
rootWindow.minsize(miniW, miniH)
rootWindow.title('Gobang')
setWindowSize(rootWindow, miniW+200, miniH)
refreshBoard()

cv.pack(fill=tkinter.BOTH, expand=1)
rootWindow.mainloop()