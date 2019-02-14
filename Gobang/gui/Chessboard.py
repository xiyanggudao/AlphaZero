from Gobang.gui.ChessboardPainter import ChessboardPainter
from Gobang.gui.PositionCalculator import PositionCalculator

class Chessboard:

	def __init__(self, canvas):
		canvas.bind('<Configure>', self.__onResize)
		canvas.bind('<Button-1>', self.__onLButtonClick)

		self.__painter = ChessboardPainter(canvas)
		self.__painter.setBoardColor('grey')

		self.__posCalculator = PositionCalculator()
		self.__posCalculator.setChessmanSpacing(3)

		self.__chessmenOnBoard = []

		self.__onClickListener = None

	def __positionAtScreen(self, x, y):
		# 棋盘坐标左下角为原点，屏幕坐标左上角为原点，需要转换
		y = 18-y
		return self.__posCalculator.positionAtScreen(x, y)

	def __positionAtBoard(self, x, y):
		pos = self.__posCalculator.positionAtBoard(x, y)
		if pos:
			return (pos[0], 18-pos[1])

	def __onResize(self, event):
		self.__posCalculator.setChessboardSize(event.width, event.height)
		self.__posCalculator.setMargin(self.__posCalculator.chessmanSize()//2+5)

		self.__painter.setChessSize(self.__posCalculator.chessmanSize())
		self.refresh()

	def __onLButtonClick(self, event):
		pos = self.__positionAtBoard(event.x, event.y)
		if self.__onClickListener:
			self.__onClickListener(pos)

	def setMoveEventListener(self, listener):
		self.__onClickListener = listener

	def __drawBackground(self):
		width, height = self.__posCalculator.boardSize()
		self.__painter.clearBoard(width, height)

	def __drawBorder(self):
		x, y = self.__posCalculator.borderPos()
		width, height = self.__posCalculator.borderSize()
		self.__painter.drawRectangle(x, y, width, height, 2)

	def __drawGrid(self):
		for row in range(1, 18):
			x1,y1 = self.__positionAtScreen(0, row)
			x2,y2 = self.__positionAtScreen(18, row)
			self.__painter.drawLine(x1,y1,x2,y2,1)
		for col in range(1, 18):
			x1, y1 = self.__positionAtScreen(col, 0)
			x2, y2 = self.__positionAtScreen(col, 18)
			self.__painter.drawLine(x1, y1, x2, y2, 1)

	def __drawPoint(self):
		x1,y1 = self.__positionAtScreen(3,3)
		x2,y2 = self.__positionAtScreen(3,15)
		x3,y3 = self.__positionAtScreen(15,3)
		x4,y4 = self.__positionAtScreen(15,15)
		x5,y5 = self.__positionAtScreen(9,9)
		radius = 5
		self.__painter.drawDisk(x1,y1, radius)
		self.__painter.drawDisk(x2,y2, radius)
		self.__painter.drawDisk(x3,y3, radius)
		self.__painter.drawDisk(x4,y4, radius)
		self.__painter.drawDisk(x5,y5, radius)

	def __drawChessman(self, pos, activeColor, step):
		chessColors = ['#000000', '#ffffff']
		color = chessColors[activeColor]
		oppositeColor = chessColors[activeColor ^ 1]
		self.__painter.setChessColor(color)
		x, y = self.__positionAtScreen(pos[0], pos[1])
		self.__painter.drawChess(x, y)
		self.__painter.drawText(x, y, str(step), oppositeColor)

	def __drawChessmen(self):
		color = 0
		for i in range(len(self.__chessmenOnBoard)):
			self.__drawChessman(self.__chessmenOnBoard[i], color, i+1)
			color ^= 1

	def minimumSize(self):
		return self.__posCalculator.boardSizeForFixedChessmanSize(30)

	def setChessmenOnBoard(self, chessmenOnBoard):
		self.__chessmenOnBoard = chessmenOnBoard

	def refresh(self):
		self.__drawBackground()
		self.__drawBorder()
		self.__drawGrid()
		self.__drawPoint()
		self.__drawChessmen()

	def printValue(self, text, pos=None):
		if pos:
			x, y = self.__positionAtScreen(pos[0], pos[1])
		else:
			x = y = self.__posCalculator.chessmanSize()
		self.__painter.drawText(x, y, text, '#ffffff')
