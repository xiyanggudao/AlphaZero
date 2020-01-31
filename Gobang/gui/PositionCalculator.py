
class PositionCalculator:

	def __init__(self, row, column):
		self.__width = 0
		self.__height = 0
		self.__margin = 0
		self.__chessmanSpacing = 0

		self.row = row
		self.column = column

	def borderPos(self):
		borderSize = self.__borderSize(self.chessmanSize())
		extraWidth = self.__width - borderSize[0]
		extraHeight = self.__height - borderSize[1]
		return (extraWidth // 2 , extraHeight // 2)

	def positionAtScreen(self, x, y):
		retX, retY = self.borderPos()
		cellSize = self.chessmanSize() + self.__chessmanSpacing
		retX += cellSize * x
		retY += cellSize * y
		return (retX, retY)

	def positionAtBoard(self, x, y):
		originX, originY = self.borderPos()
		x -= originX
		y -= originY
		chessmanSize = self.chessmanSize()
		radius = chessmanSize//2
		cellSize = chessmanSize + self.__chessmanSpacing
		x += radius
		y += radius
		if x < 0 or y < 0:
			return
		retX = x//cellSize
		retY = y//cellSize
		if x-retX*cellSize > chessmanSize or y-retY*cellSize > chessmanSize:
			return
		if retX>self.column-1 or retY>self.row-1:
			return
		return (retX, retY)

	def chessmanSize(self):
		maxWidth = self.__width
		maxWidth -= 2*self.__margin
		maxWidth -= (self.column-1)*self.__chessmanSpacing
		maxWidth //= self.column-1
		maxHeight  = self.__height
		maxHeight -= 2*self.__margin
		maxHeight -= (self.row-1)*self.__chessmanSpacing
		maxHeight //= self.row
		return min(maxWidth, maxHeight)

	def borderSize(self):
		return self.__borderSize(self.chessmanSize())

	def boardSize(self):
		return (self.__width, self.__height)

	def __borderSize(self, chessmanSize):
		retWidth, retHeight = ((self.column-1)*chessmanSize, (self.row-1)*chessmanSize)
		retWidth += (self.column-1)*self.__chessmanSpacing
		retHeight += (self.row-1)*self.__chessmanSpacing
		return (retWidth, retHeight)

	def boardSizeForFixedChessmanSize(self, chessmanSize):
		retWidth, retHeight = self.__borderSize(chessmanSize)
		retWidth += 2*self.__margin
		retHeight += 2*self.__margin
		return (retWidth, retHeight)

	def setMargin(self, margin):
		self.__margin = margin

	def setChessboardSize(self, width, height):
		self.__width = width
		self.__height = height

	def setChessmanSpacing(self, spacing):
		self.__chessmanSpacing = spacing


