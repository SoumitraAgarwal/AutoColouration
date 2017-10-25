import random
import math
import os
import cv2

class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None

def createDir(Directory):

	if(Directory not in os.listdir('.')):
		os.mkdir(Directory)


def createPatches(patchSize, trainDirectory, patchDirectory, pathchesImage):
	images = os.listdir(trainDirectory)
	for image in images:

		print("Creating patch for image " + image)
		createDir(patchDirectory)
		img 	= cv2.imread(trainDirectory + "/" + image) 

		for patch in range(pathchesImage):

			x 	= random.randint(patchSize, img.shape[0] - patchSize)
			y 	= random.randint(patchSize, img.shape[1] - patchSize)

			tpatch = img[y : y + patchSize, x : x + patchSize]
			cv2.imwrite(patchDirectory + '/' + image + "_" + str(patch) + ".JPEG", tpatch)


def initialiseTree():
	root = Tree() 
	root.left = Tree()
	root.right = Tree()
	return root


if __name__ == "__main__":


	patchSize		= 8
	pathchesImage	= 100
	trainDirectory 	= "Fishes_TRAIN"
	testDirectory	= "Fishes_TEST"
	patchDirectory	= "Patches"

	root = initialiseTree()
	# createPatches(patchSize, trainDirectory, patchDirectory, pathchesImage)


	