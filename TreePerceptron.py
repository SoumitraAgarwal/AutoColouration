from operator import add
import numpy as np, numpy.random
import pickle
import random
import math
import os
import cv2

class Tree(object):
    def __init__(self):
        self.left 		= None
        self.right 		= None
        self.weights 	= None

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


def initialiseTree(rweights, lweights, probab):

	root = Tree() 
	root.weights = probab
	root.left = Tree()
	root.left.weights = lweights
	root.right = Tree()
	root.right.weights = rweights
	return root

def trainTree(root, trainDirectory, channel, trainingRate):
	
	flag 	= 0
	counter = 1
	images 	= os.listdir(trainDirectory)
	for image in images:


		if(counter%1000 == 0):
			print("Trained for " + str(counter) + " patches out of " + str(len(images)))

		img 		= cv2.imread(trainDirectory + "/" + image)
		bgimg 		= cv2.imread(trainDirectory + "/" + image, 0)
		channels	= cv2.split(img)
		features 	= getFeatureImage(bgimg)
		results		= getFeatureImage(channels[channel])

		lprediction	= np.dot(root.left.weights, features)
		rprediction = np.dot(root.right.weights, features)
		counter	   	+= 1 		

		# print(root.weights)
		if(flag == 0):
			
			# print(lprediction - results[len(results)/2])
			normaliser 			= -1.0*trainingRate*(lprediction - results[len(results)/2])/counter
			normalisedAdd 		= [i*normaliser for i in features]
			root.left.weights 	= addLists(root.left.weights, normalisedAdd)
			normaliser 			= -1.0*trainingRate*(rprediction - results[len(results)/2])/counter
			normalisedAdd 		= [i*normaliser for i in features]
			root.right.weights 	= addLists(root.right.weights, normalisedAdd)
			flag 				= 1

		
		else:
			root.weights[0] 	= (1 - trainingRate*(lprediction - results[len(results)/2]))*root.weights[0]
			root.weights[1] 	= 1 - root.weights[0] 
			flag 				= 0


def testResults(testDirectory, storeModels, r, g, b):
	createDir(storeResults)
	images 	= os.listdir(testDirectory)
	counter = 1

	redTree 	= pickle.load( open( r, "rb" ) )
	greenTree 	= pickle.load( open( g, "rb" ) )
	blueTree 	= pickle.load( open( b, "rb" ) )

	for image in images:
		print("Tested for " + str(counter) + "images out of " + str(len(images)))
		counter	   	+= 1 
		bgimg 		= cv2.imread(trainDirectory + "/" + image, 0)

		result_img	= []
		for i in range(bgimg.shape[0]):

			result_row = []
			for j in range(bgimg.shape[1]):
				
				patch = getPatch(bgimg, i, j)
				features 	= getFeatureImage(patch)
				
				redlprediction		= np.dot(redTree.left.weights, features)
				redrprediction  	= np.dot(redTree.right.weights, features)
				greenlprediction	= np.dot(greenTree.left.weights, features)
				greenrprediction  	= np.dot(greenTree.right.weights, features)
				bluelprediction		= np.dot(blueTree.left.weights, features)
				bluerprediction		= np.dot(blueTree.left.weights, features)

def getPatch(img, i, j):
	if(i<patch_size/2):
		i = patchSize/2;
	if(j<patchSize/2):
		j = patchSize/2;

	if(i>img.shape[0]-patchSize/2 - 1):
		i = img.shape[0]-patchSize/2 - 1

	if(j>img.shape[1]-patchSize/2 - 1):
		j = img.shape[0]-patchSize/2 - 1

	return img[j - patchSize/2:j + patchSize/2, i - patchSize/2 : i + patchSize/2]

def getFeatureImage(img):
	return img.flatten()

def pickleRes(var, name, Directory):
	createDir(Directory)
	pickle.dump(var, open(Directory + '/' + name + ".p", "wb" ) )


def addLists(list1, list2):
	return [sum(x) for x in zip(list1, list2)]

if __name__ == "__main__":


	patchSize		= 9
	pathchesImage	= 100
	trainDirectory 	= "Fishes_TRAIN"
	testDirectory	= "Fishes_TEST"
	patchDirectory	= "Patches"
	storeResults 	= "Results"
	storeModels 	= "Models"
	lweights 		= np.random.dirichlet(np.ones(patchSize*patchSize),size=1)
	rweights 		= np.random.dirichlet(np.ones(patchSize*patchSize),size=1)
	probab 			= [0.5,0.5]
	trainingRate 	= 0.001

	redTree			= initialiseTree(rweights, lweights, probab)
	
	lweights 		= np.random.dirichlet(np.ones(patchSize*patchSize),size=1)
	rweights 		= np.random.dirichlet(np.ones(patchSize*patchSize),size=1)
	
	greenTree 		= initialiseTree(rweights, lweights, probab)
	
	lweights 		= np.random.dirichlet(np.ones(patchSize*patchSize),size=1)
	rweights 		= np.random.dirichlet(np.ones(patchSize*patchSize),size=1)
	
	blueTree 		= initialiseTree(rweights, lweights, probab)

	# createPatches(patchSize, trainDirectory, patchDirectory, pathchesImage)

	# trainTree(redTree, patchDirectory, 0, trainingRate)
	# pickleRes(redTree, 'red', storeModels)
	# trainTree(greenTree, patchDirectory, 1, trainingRate)
	# pickleRes(greenTree, 'green', storeModels)
	# trainTree(blueTree, patchDirectory, 2, trainingRate)
	# pickleRes(blueTree, 'blue', storeModels)
	testResults(testDirectory, patchSize, storeModels, 'red.p', 'green.p', 'blue.p')

	