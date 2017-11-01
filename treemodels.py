import math
import numpy as np

class Tree(object):
    def __init__(self):
        self.left 		= None
        self.right 		= None
        self.weights 	= None


def initialiseTree(lweights, rweights, lb, rb):

	M = np.zeros(len(lweights))
	M[len(lweights)/2 + 1]	= 1

	root 				= Tree()
	root.weights 		= [M, M]
	root.left 			= Tree()
	root.left.weights 	= np.append(lweights, lb)
	root.right 			= Tree()
	root.right.weights 	= np.append(rweights, rb)
	return root