import tkinter
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from AlphaZero.Network import Network
from AlphaZero.MCTS import MCTS
import Gobang.Config as config
from Gobang.gui.Chessboard import Chessboard
from Gobang.Game import Game

def setWindowSize(window, width, height):
	geometry = '%dx%d' % (width, height)
	window.geometry(geometry)

def refreshBoard():
	global board
	board.setChessmenOnBoard(game.chessBoard)
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

def networkAction():
	global game
	if game.isTerminated():
		return
	mcts.expandMaxNodes()
	Pi = mcts.Pi()
	if not len(Pi) > 0:
		return
	actionIndex = selectActionIndex(Pi)
	action(actionIndex)

def onKey(event):
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
		networkAction()
	elif code == 82: # r
		reset()

def onClick(pos):
	if pos == None:
		return
	global game
	if game.chessBoard[0][pos[0]][pos[1]] == 0\
		and game.chessBoard[1][pos[0]][pos[1]] == 0\
		and not game.isTerminated():
		action(pos[0]*19+pos[1])

rootWindow = tkinter.Tk()

cv = tkinter.Canvas(rootWindow)

network = Network(config.createNetworkConfig())
network.buildNetwork()
game = Game(network)
mcts = MCTS(game, config.createMCTSConfig())
board = Chessboard(cv)

board.setMoveEventListener(onClick)
rootWindow.bind('<Key>', onKey)

miniW, miniH = board.minimumSize()
rootWindow.minsize(miniW, miniH)

rootWindow.title('Gobang')
setWindowSize(rootWindow, miniW, miniH+5)
refreshBoard()

cv.pack(fill=tkinter.BOTH, expand=1)
rootWindow.mainloop()
