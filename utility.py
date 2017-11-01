from operator import add
import numpy as np
import pickle
import random
import math
import os

def getPatch(img, i, j, patchSize):
	if(i<patchSize/2):
		i = patchSize/2;
	if(j<patchSize/2):
		j = patchSize/2;

	if(i>img.shape[0]-patchSize/2 - 1):
		i = img.shape[0]-patchSize/2 - 1

	if(j>img.shape[1]-patchSize/2 - 1):
		j = img.shape[0]-patchSize/2 - 1

	return img[j - patchSize/2:j + patchSize/2 + 1, i - patchSize/2 : i + patchSize/2 + 1]


def getFeatureImage(img):
	resulter = img.flatten()
	return 1.0*resulter/256


def pickleRes(var, name, Directory):
	createDir(Directory)
	pickle.dump(var, open(Directory + '/' + name + ".p", "wb" ) )


def rescale(integ):
	integ = max(integ, 0)
	integ = min(integ, 255)
	return integ


def addLists(list1, list2):
	return [sum(x) for x in zip(list1, list2)]

def generateM(image_data):

	M = np.zeros(image_data[0].shape[0]*image_data[0].shape[1])
	M[image_data[0].shape[0]*image_data[0].shape[1]/2 + 1]	= 1
	return np.array(M)


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
