from autograd import grad
import numpy as np
import utility
import pickle
import cv2
import os

def loss_function(image_data, bgimg_data, root, M1, M2, channel):

	image_counter 	= 0
	leftprobs 		= []
	rightprobs		= []
	for img, bgimg in zip(image_data, bgimg_data):
		image_counter += 1
		# if(image_counter%1000 == 0):
		# 	print("Completed probability calculation over " + str(image_counter))
		
		channels	= cv2.split(img)
		features 	= utility.getFeatureImage(bgimg)
		results		= utility.getFeatureImage(channels[channel])

		lprediction	= np.add(np.dot(root.left.weights[:-1], features), root.left.weights[-1:])
		rprediction = np.add(np.dot(root.right.weights[:-1], features), root.right.weights[-1:])

		uleft 		= 1.0*lprediction
		uright 		= 1.0*rprediction

		leftprob 	= 1.0*np.exp(uleft)/(np.exp(uleft) + np.exp(uright))
		rightprob 	= 1.0*np.exp(uright)/(np.exp(uleft) + np.exp(uright))
		
		leftprobs.append(leftprob)
		rightprobs.append(rightprob)

	leftprobsum 	= sum(leftprobs)
	rightprobsum 	= sum(rightprobs)
	lossleft,lossright	= 0,0
	for leftprob,rightprob,img,bgimg in zip(leftprobs, rightprobs, image_data, bgimg_data):

		w1 			= 1.0*leftprob/leftprobsum
		w2 			= 1.0*rightprob/rightprobsum
		channels	= cv2.split(img)
		features 	= utility.getFeatureImage(bgimg)
		results		= utility.getFeatureImage(channels[channel])
		result_col 	= results[(bgimg.shape[0]/2+1)*(bgimg.shape[1]/2+1)]
		lossleft 	-= w1*(np.array(result_col) - np.dot(M1, features))*features
		lossright 	-= w2*(np.array(result_col) - np.dot(M2, features))*features

	return lossleft, lossright


def weight_loss(image_data, bgimg_data, root, M1, M2, channel):

	image_counter 	= 0
	leftprobs 		= []
	rightprobs		= []
	derivativesl	= []
	derivativesr 	= []
	for img, bgimg in zip(image_data, bgimg_data):
		
		image_counter += 1
		
		channels	= cv2.split(img)
		features 	= utility.getFeatureImage(bgimg)
		results		= utility.getFeatureImage(channels[channel])

		lprediction	= np.add(np.dot(root.left.weights[:-1], features), root.left.weights[-1:])
		rprediction = np.add(np.dot(root.right.weights[:-1], features), root.right.weights[-1:])

		uleft 		= 1.0*lprediction
		uright 		= 1.0*rprediction

		leftprob 	= 1.0*np.exp(uleft)/(np.exp(uleft) + np.exp(uright))
		rightprob 	= 1.0*np.exp(uright)/(np.exp(uleft) + np.exp(uright))
		
		mult 		= 1.0*np.exp(uleft)*np.exp(uright)/pow(np.exp(uleft) + np.exp(uright),2)
		lderiv 		= np.append(features, 1)*mult
		rderiv 		= np.append(features, 1)*mult

		derivativesl.append(lderiv)
		derivativesr.append(rderiv)

		leftprobs.append(leftprob)
		rightprobs.append(rightprob)

	print(leftprobs)
	leftprobsum 	= sum(leftprobs)
	rightprobsum 	= sum(rightprobs)
	lderivsum		= sum(derivativesl)
	rderivsum 		= sum(derivativesr)

	lossleft,lossright	= 0,0
	for leftprob,rightprob,img,bgimg, lderiv, rderiv in zip(leftprobs, rightprobs, image_data, bgimg_data, derivativesl, derivativesr):

		w1 			= 1.0*leftprob/leftprobsum
		w2 			= 1.0*rightprob/rightprobsum
		channels	= cv2.split(img)
		features 	= utility.getFeatureImage(bgimg)
		results		= utility.getFeatureImage(channels[channel])
		result_col 	= results[(bgimg.shape[0]/2+1)*(bgimg.shape[1]/2+1)]
		lderivmul 	= 1.0*lderiv/leftprobsum - 1.0*lderivsum*leftprob/(leftprobsum*leftprobsum)
		rderivmul 	= 1.0*rderiv/rightprobsum - 1.0*rderivsum*rightprob/(rightprobsum*rightprobsum)
		lossleft 	+= 0.5*lderivsum*pow((np.array(result_col) - np.dot(M1, features)),2)
		lossright 	+= 0.5*rderivsum*pow((np.array(result_col) - np.dot(M1, features)),2)
	
	return lossleft,lossright

def trainTree(root, trainDirectory, channel, trainingRate, iterations):
	
	flag 		= 0
	counter 	= 1
	image_data 	= []
	bgimg_data 	= []
	images 		= os.listdir(trainDirectory)
	images 		= np.random.choice(images, 2000)

	for image in images:

		if(counter%1000 == 0):
			print("Preparing image list, done with " + str(counter))
		img 		= cv2.imread(trainDirectory + "/" + image)
		bgimg 		= cv2.imread(trainDirectory + "/" + image, 0)
		image_data.append(img)
		bgimg_data.append(bgimg)
		counter += 1
	
	M1 			= utility.generateM(image_data)
	M2			= utility.generateM(image_data)


	for counter in range(iterations):
		print("Running iteration : " + str(counter))
		
		def left_loss_now(M):
			return loss_function(image_data, bgimg_data, root, M, M2, channel)[0]

		def right_loss_now(M):
			return loss_function(image_data, bgimg_data, root, M1, M, channel)[1]

		for k in range(0):

			left_gradient 	= left_loss_now(M1)
			M1 			  	= np.add(M1, -trainingRate*left_gradient)
			root.weights[0]	= M1
			right_gradient 	= right_loss_now(M2)
			M2 			  	= np.add(M2, -trainingRate*right_gradient)
			root.weights[1]	= M2
			print("Running gradient descent for M step : " + str(k))


		def weight_loss_left(root1):
			return weight_loss(image_data, bgimg_data, root1, M1, M2, channel)[0];

		def weight_loss_right(root1):
			return weight_loss(image_data, bgimg_data, root1, M1, M2, channel)[1];

		for k in range(10):

			# print(root.left.weights)
			left_gradient 		= weight_loss_left(root)
			# print(left_gradient)
			root.left.weights 	= np.add(root.left.weights, -0.00001*trainingRate*left_gradient)
			right_gradient 		= weight_loss_right(root)
			root.right.weights 	= np.add(root.right.weights, -0.00001*trainingRate*right_gradient)
			print("Running gradient descent for weights step : " + str(k))



def directM(root, trainDirectory, channel, image_data, bgimg_data):

	image_counter 	= 0
	leftprobs 		= []
	rightprobs		= []
	for img, bgimg in zip(image_data, bgimg_data):
		
		image_counter += 1
		
		channels	= cv2.split(img)
		features 	= utility.getFeatureImage(bgimg)
		results		= utility.getFeatureImage(channels[channel])

		lprediction	= np.add(np.dot(root.left.weights[:-1], features), root.left.weights[-1:])
		rprediction = np.add(np.dot(root.right.weights[:-1], features), root.right.weights[-1:])

		uleft 		= 1.0*lprediction
		uright 		= 1.0*rprediction

		leftprob 	= 1.0*np.exp(uleft)/(np.exp(uleft) + np.exp(uright))
		rightprob 	= 1.0*np.exp(uright)/(np.exp(uleft) + np.exp(uright))
		
		leftprobs.append(leftprob)
		rightprobs.append(rightprob)

	sumleftprobs = sum(leftprobs)
	sumrightprobs = sum(rightprobs)
	leftprobs 	= [pow(1.0*i/sumleftprobs, 0.5)[0] for i in leftprobs]
	rightprobs 	= [pow(1.0*i/sumrightprobs, 0.5)[0] for i in rightprobs]

	O1 	= np.diag(leftprobs)
	O2	= np.diag(rightprobs)

	M1	= root.weights[0]
	M2 	= root.weights[1]
	
	X 	= []
	Y 	= []
	for leftprob,rightprob,img,bgimg in zip(leftprobs, rightprobs, image_data, bgimg_data):

		channels	= cv2.split(img)
		features 	= utility.getFeatureImage(bgimg)
		results		= utility.getFeatureImage(channels[channel])
		result_col 	= results[(bgimg.shape[0]/2+1)*(bgimg.shape[1]/2+1)]
		Y.append(np.array(result_col))
		X.append(features)

	Y = np.asarray(Y)
	X = np.asarray(X)

	M1	= np.dot(np.dot(O1,Y).T, np.dot(O1,X))/np.dot(np.dot(O2,Y).T, np.dot(O2,Y))
	print(M1.shape)
	root.weights[0] = M1
	M2	= np.dot(np.dot(np.linalg.inv(O2), np.dot(np.dot(O2,Y), np.dot(O2,X).T)/np.dot(np.dot(O2,Y).T, np.dot(O2,Y))), np.inv(O2))
	root.weights[1] = M2

def trainTreeDirect(root, trainDirectory, channel, trainingRate, iterations):
	
	flag 		= 0
	counter 	= 1
	image_data 	= []
	bgimg_data 	= []
	images 		= os.listdir(trainDirectory)
	images 		= np.random.choice(images, 100)

	for image in images:

		if(counter%1000 == 0):
			print("Preparing image list, done with " + str(counter))
		img 		= cv2.imread(trainDirectory + "/" + image)
		bgimg 		= cv2.imread(trainDirectory + "/" + image, 0)
		image_data.append(img)
		bgimg_data.append(bgimg)
		counter += 1
	
		
	directM(root, trainDirectory, channel, image_data, bgimg_data)
	print("Updated M")


	for counter in range(iterations):

		def weight_loss_left(root1):
			return weight_loss(image_data, bgimg_data, root1, M1, M2, channel)[0];

		def weight_loss_right(root1):
			return weight_loss(image_data, bgimg_data, root1, M1, M2, channel)[1];

		for k in range(10):

			# print(root.left.weights)
			left_gradient 		= weight_loss_left(root)
			# print(left_gradient)
			root.left.weights 	= np.add(root.left.weights, -0.00001*trainingRate*left_gradient)
			right_gradient 		= weight_loss_right(root)
			root.right.weights 	= np.add(root.right.weights, -0.00001*trainingRate*right_gradient)
			print("Running gradient descent for weights step : " + str(k))









def testResults(testDirectory, patchSize, storeModels, storeResults, r, g, b):
	utility.createDir(storeResults)
	images 	= os.listdir(testDirectory)
	counter = 1

	redTree 	= pickle.load( open(storeModels + '/' + r, "rb" ) )
	greenTree 	= pickle.load( open(storeModels + '/' + g, "rb" ) )
	blueTree 	= pickle.load( open(storeModels + '/' + b, "rb" ) )

	for image in images:
		print("Tested for " + str(counter) + " images out of " + str(len(images)))
		counter	   	+= 1 
		bgimg 		= cv2.imread(testDirectory + "/" + image, 0)
		img 		= cv2.imread(testDirectory + "/" + image)
		result_img 		 = []
		for i in range(bgimg.shape[0]):

			result_row 			= []
			for j in range(bgimg.shape[1]):
				
				patch 		= utility.getPatch(bgimg, i, j, patchSize)
				features 	= utility.getFeatureImage(patch)
				
				redlprediction		= np.add(np.dot(redTree.left.weights[:-1], features), redTree.left.weights[-1:])
				redrprediction  	= np.add(np.dot(redTree.right.weights[:-1], features), redTree.right.weights[-1:])
				greenlprediction	= np.add(np.dot(greenTree.left.weights[:-1], features), greenTree.left.weights[-1:])
				greenrprediction  	= np.add(np.dot(greenTree.right.weights[:-1], features), greenTree.right.weights[-1:])
				bluelprediction		= np.add(np.dot(blueTree.left.weights[:-1], features), blueTree.left.weights[-1:])
				bluerprediction		= np.add(np.dot(blueTree.right.weights[:-1], features), blueTree.right.weights[-1:])
				
				pred 	= 1.0*redlprediction/(redlprediction + redrprediction)
				pblue 	= 1.0*bluelprediction/(bluelprediction + bluerprediction)
				pgreen 	= 1.0*greenlprediction/(greenlprediction + greenrprediction)
				
				red 	= utility.rescale(255*(pred*np.dot(redTree.weights[0], features) + (1 - pred)*np.dot(redTree.weights[1], features)))
				green 	= utility.rescale(255*(pgreen*np.dot(greenTree.weights[0], features) + (1 - pgreen)*np.dot(greenTree.weights[1], features)))
				blue 	= utility.rescale(255*(pblue*np.dot(blueTree.weights[0], features) + (1 - pblue)*np.dot(blueTree.weights[1], features)))
				
				pixer 	= [int(red[0]), int(green[0]), int(blue[0])]
				result_row.append(pixer)

			result_img.append(result_row)

		result_img 		 	= np.array(result_img)
		# print(result_img)		
		cv2.imwrite( storeResults + "/" + image, result_img)
