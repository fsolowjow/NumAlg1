from __future__ import division
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA

def heavyside(x):
	if(	x >= 0):				return 1
	else:						return 0
	

def Riemann_2D_solve( A , q_l, q_r):
	eigenvalue , eigenvector = LA.eig(A)
	alpha = LA.solve(eigenvector, q_l - q_r)
	w1 = alpha[0] * eigenvector[0] 
	w2 = alpha[1] * eigenvector[1]
	time = 0.5
	x = np.arange(-1, 3 , 0.1)
	y = np.arange(-1, 3 , 0.1)
	solution1 = np.zeros(np.size(x*y))
	solution2 = np.zeros(np.size(x*y))
	
	for j in range(np.size(y)):
		for i in range(np.size(x)):
			solution1[i] = (q_l + ( heavyside( x[i] - eigenvalue[0] * time) * w1 ) + ( heavyside( y[j] - eigenvalue[1] * time) * w2 ) )[0]
			solution2[i] = (q_l + ( heavyside( x[i] - eigenvalue[0] * time) * w1 ) + ( heavyside( y[j] - eigenvalue[1] * time) * w2 ) )[1]

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(np.outer( x , np.ones(np.size(x))) , y , np.split(solution1,np.size(x)), rstride=4, cstride=4, color='b')
	ax.plot_surface(np.outer( x , np.ones(np.size(x))) , y , np.split(solution2,np.size(x)), rstride=4, cstride=4, color='r')
	plt.show()
	
A1 = np.array ( [[2 , 1] , [0.0001,2]])
q_l1 = np.array([0,1])
q_r1 = np.array([1,0])
Riemann_2D_solve( A1, q_l1, q_r1)

A2 = np.array ( [[1 , 1] , [1 , 1]])
q_l2 = np.array([1,0])
q_r2 = np.array([2,0])
Riemann_2D_solve( A2, q_l2 , q_r2)


