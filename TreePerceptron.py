import math
import cv2


patchSize		= 10
trainDirectory 	= "Fishes_TRAIN"
testDirectory	= "Fishes_TEST"
patchDirectory	= "Patches"



class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None

def createPatches(patchSize, trainDirectory):




def initialiseTree():
	root = Tree() 
	root.left = Tree()
	root.right = Tree()
	return root


if __name__ == "__main__":
	root = initialiseTree()
	createPatches(patchSize, trainDirectory)