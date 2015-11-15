from __future__ import division
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
	
def initial_cond1(x):
	return np.exp(-4 * (x-1)*(x-1))

def initial_cond2(x):
	if( isinstance(x, float) == True or isinstance(x, int) == True):
		if( x > 0 ):	return 0
		else:			return 1
	else:
		temp = np.zeros(np.size(x))
		for i in range(np.size(x)):
			if( x[i] <= 0 ):	temp[i]=1
		return temp
	
def initial_cond3(x):
	if( isinstance(x, float) == True or isinstance(x, int) == True):
		if( x <= 0):			return 0
		if( x < 1 and x > 0):	return (1-x)
		if(	x >= 1):			return 1
	else:
		temp = np.zeros(np.size(x))
		for i in range(np.size(x)):
			if( x[i] < 1 and x[i] > 0):	temp[i] = (1-x[i])
			if(	x[i] >= 1):				temp[i] = 1
		return temp
	
def initial_cond (x,i):
	if(i==1): return initial_cond1(x)
	if(i==2): return initial_cond2(x)
	if(i==3): return initial_cond3(x)	

		
def plot_char(y_l, y_r, t_i , t_e):
	ini_cond = 1											# initial conditions \in {1,2,3} 
	num_points = 30
	y = np.zeros(num_points)
	values_char = np.zeros(num_points*num_points)
	values_solution = np.zeros(num_points*num_points)
	zeiten = np.zeros(num_points)
	
	for i in range(num_points):
		y[i] = y_l + ((y_r - y_l)*(i/num_points))
	for i in range(num_points):
		zeiten[i] = t_i + ((t_e - t_i) * i/num_points)
	for j in range(num_points):
		values_char[j*num_points : (j+1)*num_points] = y[j] + zeiten * initial_cond(y[j], ini_cond)
		values_solution[j*num_points : (j+1)*num_points] = initial_cond(y + zeiten[j] * initial_cond(y,ini_cond) , ini_cond)
	for j in range(num_points):
		plt.plot(zeiten , values_char[j*num_points : (j+1)*num_points])
	plt.show()
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(np.outer(zeiten,np.ones(num_points)) , y , np.split(values_solution,num_points), rstride=4, cstride=4, color='b')
	plt.show()

plot_char(-1. , 3. , 0. , 2.)	

