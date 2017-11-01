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
		if(image_counter%1000 == 0):
			print("Completed probability calculation over " + str(image_counter))
		
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
		lossleft 	+= w1*pow((result_col - np.dot(M1, features)),2)
		lossright 	+= w2*pow((result_col - np.dot(M2, features)),2)

	return lossleft, lossright

def trainTree(root, trainDirectory, channel, trainingRate, iterations):
	
	flag 		= 0
	counter 	= 1
	image_data 	= []
	bgimg_data 	= []
	images 		= os.listdir(trainDirectory)
	images 		= images[:3000]

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

		for k in range(100):

			print(k)
			left_gradient 	= grad(left_loss_now)
			M1 			  	= [i - trainingRate*left_gradient(M1) for i in M1]
			root.weights[0]	= M1
			right_gradient 	= grad(right_loss_now)
			M2 			  	= [i - trainingRate*right_gradient(M2) for i in M2]
			root.weights[1]	= M2

			print(M1[0])


def testResults(testDirectory, patchSize, storeModels, storeResults, r, g, b):
	createDir(storeResults[0])
	createDir(storeResults[1])
	createDir(storeResults[2])
	images 	= os.listdir(testDirectory)
	counter = 1

	redTree 	= pickle.load( open(storeModels + '/' + r, "rb" ) )
	greenTree 	= pickle.load( open(storeModels + '/' + g, "rb" ) )
	blueTree 	= pickle.load( open(storeModels + '/' + b, "rb" ) )

	for image in images:
		print("Tested for " + str(counter) + " images out of " + str(len(images)))
		counter	   	+= 1 
		bgimg 		= cv2.imread(trainDirectory + "/" + image, 0)
		img 		= cv2.imread(trainDirectory + "/" + image)
		result_img_red	 = []
		result_img_blue	 = []
		result_img_green = []
		result_img 		 = []
		for i in range(bgimg.shape[0]):

			result_row_red 		= []
			result_row_blue 	= []
			result_row_green 	= []
			result_row 			= []
			for j in range(bgimg.shape[1]):
				
				patch 		= getPatch(bgimg, i, j, patchSize)
				features 	= getFeatureImage(patch)
				
				redlprediction		= np.dot(redTree.left.weights, features)
				redrprediction  	= np.dot(redTree.right.weights, features)
				greenlprediction	= np.dot(greenTree.left.weights, features)
				greenrprediction  	= np.dot(greenTree.right.weights, features)
				bluelprediction		= np.dot(blueTree.left.weights, features)
				bluerprediction		= np.dot(blueTree.right.weights, features)
				red 	= rescale(redlprediction*redTree.weights[0] + redrprediction*redTree.weights[1])
				green 	= rescale(greenlprediction*greenTree.weights[0] + greenrprediction*greenTree.weights[1])
				blue 	= rescale(bluelprediction*blueTree.weights[0] + bluerprediction*blueTree.weights[1])
				
				red, blue, green = bgimg[i][j],bgimg[i][j],bgimg[i][j]
				pixel_val = [red, 0, 0]
				result_row_red.append(pixel_val)
				pixel_val = [0, green, 0]
				result_row_green.append(pixel_val)
				pixel_val = [0, 0, blue]
				result_row_blue.append(pixel_val)

				pixer = [rescale(int(img[i][j][0]*1.0*random.randint(1,27)/7)), rescale(int(img[i][j][1]*1.0*random.randint(1,27)/7)), rescale(int(img[i][j][2]*1.0*random.randint(1,27)/7))]
				result_row.append(pixer)
			result_img_red.append(result_row_red)
			result_img_green.append(result_row_green)
			result_img_blue.append(result_row_blue)
			result_img.append(result_row)

		result_img_red 		= np.array(result_img_red)
		result_img_green 	= np.array(result_img_green)
		result_img_blue 	= np.array(result_img_blue)
		result_img 		 	= np.array(result_img)
		
		cv2.imwrite(storeResults[0] + '/' + image, result_img_red)
		cv2.imwrite(storeResults[1] + '/' + image, result_img_green)
		cv2.imwrite(storeResults[2] + '/' + image, result_img_blue)
		cv2.imwrite("Results_Combined/" + image, result_img)
