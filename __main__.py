import numpy as np, numpy.random
import treemodels
import utility
import neural
import cv2

if __name__ == "__main__":


	patchSize		= 9
	pathchesImage	= 100
	iterations 		= 30
	trainDirectory 	= "Fishes_TRAIN"
	testDirectory	= "Fishes_TEST"
	patchDirectory	= "Patches"
	storeResults 	= "Results"
	storeModels 	= "Models"
	lweights 		= [np.random.uniform(0,1,patchSize*patchSize)]
	rweights 		= [np.random.uniform(0,1,patchSize*patchSize)]
	# lweights 		= np.random.dirichlet(np.ones(patchSize*patchSize),size=1)
	# rweights 		= np.random.dirichlet(np.ones(patchSize*patchSize),size=1)
	lb 				= lweights[0][0]*rweights[0][0]
	rb 				= rweights[0][0]*lweights[0][0]
	trainingRate 	= 0.1

	redTree			= treemodels.initialiseTree(lweights[0], rweights[0], lb, rb)
	greenTree 		= treemodels.initialiseTree(lweights[0], rweights[0], lb, rb)
	blueTree 		= treemodels.initialiseTree(lweights[0], rweights[0], lb, rb)

	# utility.createPatches(patchSize, trainDirectory, patchDirectory, pathchesImage)

	neural.trainTreeDirect(redTree, patchDirectory, 0, trainingRate, iterations)
	utility.pickleRes(redTree, 'redDirect', storeModels)
	neural.trainTree(greenTree, patchDirectory, 1, trainingRate, iterations)
	utility.pickleRes(greenTree, 'greenDirect', storeModels)
	neural.trainTree(blueTree, patchDirectory, 2, trainingRate, iterations)
	utility.pickleRes(blueTree, 'blueDirect', storeModels)
	neural.testResults(testDirectory, patchSize, storeModels, storeResults, 'redDirect.p', 'greenDirect.p', 'blueDirect.p')

	