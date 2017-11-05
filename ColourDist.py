import cv2
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

train = 'Fishes_TEST'
test  = 'Fishes_TEST'

train_images = os.listdir(train)
train_images = np.random.choice(train_images, 20)
colours		= []
intensity 	= []
for image in train_images:

	print(image)
	img 	= cv2.imread(train + '/' + image)
	bgimg 	= cv2.imread(train + '/' + image, 0)
	bgimg 	= bgimg.flatten()

	colours		+= img.reshape(64*64, -1).tolist()
	intensity 	+= bgimg.tolist()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

iterator = 0
for values in colours:
	print(iterator, len(colours))
	ax.scatter(values[0], values[1], values[2], c=[i/255. for i in values])
	iterator += 1

ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')

plt.show()
