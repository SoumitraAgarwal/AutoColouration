import numpy as np, numpy.random
import treemodels
import utility
import neural
import cv2

if __name__ == "__main__":


	patchSize		= 9
	pathchesImage	= 100
	iterations 		= 100
	trainDirectory 	= "Fishes_TRAIN"
	testDirectory	= "Fishes_TEST"
	patchDirectory	= "Patches"
	storeResults 	= ["Results_Blue", "Results_Green", "Results_Red"]
	storeModels 	= "Models"
	lweights 		= np.random.dirichlet(np.ones(patchSize*patchSize),size=1)
	rweights 		= np.random.dirichlet(np.ones(patchSize*patchSize),size=1)
	lb 				= 1
	rb 				= 1
	trainingRate 	= 0.01

	redTree			= treemodels.initialiseTree(lweights[0], rweights[0], lb, rb)
	greenTree 		= treemodels.initialiseTree(lweights[0], rweights[0], lb, rb)
	blueTree 		= treemodels.initialiseTree(lweights[0], rweights[0], lb, rb)

	# utility.createPatches(patchSize, trainDirectory, patchDirectory, pathchesImage)

	neural.trainTree(redTree, patchDirectory, 0, trainingRate, iterations)
	utility.pickleRes(redTree, 'red', storeModels)
	trainTree(greenTree, patchDirectory, 1, trainingRate)
	pickleRes(greenTree, 'green', storeModels)
	trainTree(blueTree, patchDirectory, 2, trainingRate)
	pickleRes(blueTree, 'blue', storeModels)
	# neural.testResults(testDirectory, patchSize, storeModels, storeResults, 'red.p', 'green.p', 'blue.p')

	