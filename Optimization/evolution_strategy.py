import numpy as np
from matplotlib import pyplot as plt
from random import choice, random
import math

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm



def eggholder(x1,x2,x3,x4,x5):
	summa = 0
	xin = [x1,x2,x3,x4,x5]
	for i in range(len(xin)-1):
		val = (-(xin[i+1] + 47)*np.sin(np.sqrt(np.abs(xin[i+1]+(xin[i]/2)+47))) - xin[i] *np.sin(np.sqrt(np.abs(xin[i] - (xin[i+1]+47)))))
		summa += val
	return summa

def initi(mu, n, limits):
	#This function initializes x0 and sigma. Also, copmutes the rotation angles matrix alpha
	#returns x0, sigma, alpha

	#Generate random parent population:
	x0 = np.zeros([n,mu])
	for i in range(n):
		x0[i,:] = np.random.uniform(-512,512,mu)

	# Generate covariance matrix
	sigma = []
	for i in range(mu):
		tmp = np.random.uniform(size=(n,n))
		sigma.append(tmp + np.transpose(tmp))

	#Generate rotation agnel matrices
	alpha = []
	for i in range(mu):
		alpha_array = np.zeros([n,n])
		for j in range(n-1):
			for k in range(j+1,n):
				alpha_array[j,k] = 0.5*math.atan2(2*sigma[i][j,k],(sigma[i][j,j]**2 - sigma[i][k,k]**2))
		alpha.append(alpha_array)

	return {'x0':x0, 'sigma':sigma, 'alpha':alpha}


def recombination(type_obj,type_str,n,mu,lambd,xi,sigma,alpha):
	#This function performs recombination. It has 5 different recombinations to pick choose from

	xip    = np.zeros([n,lambd]);
	sigmap = []
	alphap = [] 

	# Recombination of object variables:

	#No recombination
	if type_obj == 1:
	    for i in range(lambd):
	      idx = int(np.random.randint(1,mu,1))
	      xip[:,i]  = xi[:,idx];

	 #Discrete recombination
	elif type_obj == 2:
		for i in range(lambd):
			tmp = (np.random.randint(1,mu,2)) #pick two parents
			for j in range(n): 
				idx = np.random.choice(tmp) 
				xip[j,i]  = xi[j,idx];  

	#Global discrete recombination
	elif type_obj == 3:
		i = 0
		while i < lambd:
			fixed = int(np.random.randint(1,mu,1))
			for k in range(mu):
				for j in range(n):
					tmp = int(np.random.randint(1,mu,1))
					idx = np.random.choice([fixed,tmp])
					xip[j,i]  = xi[j,idx]
			i = i+1

	#Intermediate recombination
	elif type_obj == 4:
		for i in range(lambd):
			tmp = (np.random.randint(1,mu,2))
	 		xip[:,i] = xi[:,tmp[0]] + (xi[:,tmp[1]] - xi[:,tmp[0]])/2 

	#Global intermediate recombination
	elif type_obj == 5:
		i = 0
		while i < lambd:
			fixed = int(np.random.randint(1,mu,1))
			for k in range(mu):
				for j in range(n):
					tmp = int(np.random.randint(1,mu,1))
					xip[:,i] = xi[j,fixed] + (xi[j,tmp] - xi[j,fixed])/2
			i = i+1

	# We now perform recombination for the strategy parameters:
	#no recombination
	if type_str == 1:
		for i in range(lambd):
			idx = int(np.random.randint(1,mu,1))
			sigmap.append(sigma[idx])
			alphap.append(alpha[idx])
 #Discrete recombination
	elif type_str == 2:
		for i in range(lambd):
			tmp = (np.random.randint(1,mu,2))
			temp_sigma = np.zeros([n,n])
			alpha_temp = np.zeros([n,n])
			for j in range(n):
				for jj in range(n):
					idx = np.random.choice(tmp)
					temp_sigma[j,jj] = sigma[idx][j,jj]
					temp_sigma[jj,j] = sigma[idx][j,jj]
					alpha_temp[j,jj] = alpha[idx][j,jj]
			sigmap.append(temp_sigma)
			alphap.append(alpha_temp)

	#Global discrete recombination
	elif type_str == 3:
		i = 0
		while i < lambd:
			fixed = int(np.random.randint(1,mu,1))
			for k in range(mu):
				temp_sigma = np.zeros([n,n])
				alpha_temp = np.zeros([n,n])
				for j in range(n):
					for jj in range(n):
						tmp = int(np.random.randint(1,mu,1))
						idx = np.random.choice([fixed,tmp])
						temp_sigma[j,jj] = sigma[idx][j,jj]
						temp_sigma[jj,j] = sigma[idx][j,jj]
						alpha_temp[j,jj] = alpha[idx][j,jj]
				sigmap.append(temp_sigma)
				alphap.append(alpha_temp)
			i = i+1	

	# Intermediate recombination
	elif type_str == 4:		
		for i in range(lambd):
			tmp =(np.random.randint(1,mu,2))
			new_sigma = sigma[tmp[0]] + (sigma[tmp[1]] - sigma[tmp[0]])/2
			new_alpha = alpha[tmp[0]] + (alpha[tmp[1]] - alpha[tmp[0]])/2
			# validate rotation angles (they must be between [-pi, pi]
			[p,m] = np.nonzero(abs(new_alpha) > math.pi)
			new_alpha[p,m] =  new_alpha[p,m] - 2*math.pi*(new_alpha[p,m]/abs(new_alpha[p,m]))
			#validate standard deviations (must be greater than zero)
			[p,m] = np.nonzero(new_sigma <= 0)
			new_sigma[p,m] = 0.1
			sigmap.append(new_sigma)
			alphap.append(new_alpha)

	#Global intermediate recombination
	elif type_str == 5:
		i = 0
		while i < lambd:
			fixed = int(np.random.randint(1,mu,1))
			for k in range(mu):
				temp_sigma = np.zeros([n,n])
				alpha_temp = np.zeros([n,n])
				for j in range(n):
					for jj in range(n):
						tmp = int(np.random.randint(1,mu,1))
						temp_sigma[j,jj]  = sigma[fixed][j,jj] + (sigma[tmp][j,jj] - sigma[fixed][j,jj])/2
						temp_sigma[jj,j] = sigma[fixed][j,jj] + (sigma[tmp][j,jj] - sigma[fixed][j,jj])/2
						alpha_temp[j,jj] = alpha[fixed][j,jj] + (alpha[tmp][j,jj] - alpha[fixed][j,jj])/2
				sigmap.append(temp_sigma)
				alphap.append(alpha_temp)
				# validate rotation angles (they must be between [-pi, pi])
				[p,m] = np.nonzero(abs(alphap[i]) > math.pi)
				alphap[i][p,m] = alphap[i][p,m] - 2*math.pi*(alphap[i][p,m]/abs(alphap[i][p,m]))
				#validate standard deviations (must be greater than zero)
				[p,m] = np.nonzero(sigmap[i] <= 0)
				sigmap[i][p,m] = 0.1
			i = i+1

	return {'xip':xip, 'sigmap':sigmap, 'alphap': alphap}

def mutation(n,lambd,xr,sigmap,alphap,limits):
	#This function performs mutation, as described in the lecture notes.

	#learning rate
	tau   = 1/(np.sqrt(2*np.sqrt(n)));
	# learning rate
	taup  = 1/(np.sqrt(2*n)); 
	# 5 degrees (in radians)                  
	beta  = 5*math.pi/180;                    

	#Mutate:
	xm     = np.zeros([n,lambd]);
	sigmam =[]
	alpham = []

	for i in range (lambd):
		tmp = np.random.standard_normal(size = [n,n])
		sigmam.append(np.multiply(sigmap[i], np.exp(taup*float(np.random.standard_normal(1))+tau*(tmp+np.transpose(tmp)))))
		tmp = np.random.uniform(size = [n,n])
		alpham.append(alphap[i]+ beta*np.triu(tmp+np.transpose(tmp),1))

	#   % Coordinate transformation with respect to axes 'i' and 'j' and angle
		R = np.eye(n,n)
		for m in range(n-1):
			for q in range(m+1,n):
				T = np.eye(n,n)
				T[m,m] = np.cos(alpham[i][m,m])
				T[m,q] = -np.sin(alpham[i][m,q])
				T[q,m] = np.sin(alpham[i][q,m])
				T[q,q] = np.cos(alpham[i][q,q])
				R = np.dot(R,T)

		mult1 = np.dot(R, np.sqrt(np.diag(np.diag(sigmam[i]))))
		mult2 = np.dot(mult1, np.random.standard_normal(size=(n)))
		xm[:,i] = xr[:,i] + mult2

		#Check if constraints are being violated:
		for ii in range(n):
			if xm[ii,i] < limits[ii,0]:
				xm[ii,i] = limits[ii,0]
			elif xm[ii,i] > limits[ii,1]:
				xm[ii,i] = limits[ii,1]
			else:
				continue

	return {'xm': xm, 'sigmam': sigmam, 'alpham':alpham}

def selection(scheme, mu, lambd, epse, eps, xm, sigmam, x0, sigma, alpham, alpha):
	#This function performs selection. Choose between (mu,lambda)-selection or (mu+lambda)-selection

	#(mu,lambda) selection
	if scheme == ',':
			if (mu > lambd):
				err = epse
				idx = np.argsort(err)
				idx = idx[0:mu]
				min_x = xm[:,idx.tolist()]
				min_sigma = [sigmam[x] for x in idx.tolist()]
				min_alpha = [alpham[x] for x in idx.tolist()]

	#(mu+lambda) selection
	elif scheme == '+':
		err = np.concatenate((epse,eps))
		xaug = np.concatenate((xm, x0), 1)
		sigmaaug = sigmam + sigma
		alphaaug = alpham + alpha
		idx = np.argsort(err)
		idx = idx[0:mu]
		min_x = xaug[:,idx.tolist()]
		min_sigma = [sigmaaug[x] for x in idx.tolist()]
		min_alpha = [alphaaug[x] for x in idx.tolist()]

	return {'min_x':min_x, 'min_sigma': min_sigma, 'min_alpha': min_alpha}


def evolution_strategy(mu,lambd,gen,sel,rec_obj,rec_str,obj,nf,n,limits):
	#We now use all of the functions defined to perform evolution strategy.


	#Start by initializing our object variables and strategy parameters
	val_size_dict = initi(mu, n, limits)
	x0 = val_size_dict['x0']
	sigma = val_size_dict['sigma']
	alpha = val_size_dict['alpha']

	min_x = []                            # min variables
	min_f = []                  		  # min values
	off   = []                  		  # offsprings

	#Evaluate at first data point
	min_x.append(x0)                   
	value = np.zeros((nf,mu))            
	for i in range(mu):
		value[:,i] = eggholder(x0[0,i],x0[1,i],x0[2,i],x0[3,i],x0[4,i])
	min_f.append(value)                   
	off.append(np.zeros((n,1)))

	j      = 0;                           # generations counter
	eps    = abs(obj - value[0,:]);       # initial error
	EPS    = np.zeros([gen,1]);           # matrix to store all errors
	EPS[0] = min(eps);

	#begin ES
	while ((j < gen)):
		#recombine
		recombine_dict = recombination(rec_obj, rec_str, n, mu, lambd, min_x[j], sigma, alpha)
		xr = recombine_dict['xip']
		sigmar = recombine_dict['sigmap']
		alphar = recombine_dict['alphap']
		off.append(xr)

		#mutation
		mutation_dict = mutation(n, lambd, xr, sigmar, alphar, limits)
		xm = mutation_dict['xm']
		sigmam = mutation_dict['sigmam']
		alpham = mutation_dict['alpham']


		#evaluation
		phie = np.zeros((nf,lambd))

		#Evaluate error
		for i in range(lambd):
			phie[:,i] = eggholder(xm[0,i],xm[1,i],xm[2,i],xm[3,i],xm[4,i])
		epse = abs(obj-phie[0,:])

		#Selection
		selection_dict = selection(sel, mu, lambd, epse, eps, xm, sigmam, min_x[j], sigma, alpham, alpha)
		min_x.append(selection_dict['min_x'])
		sigma = selection_dict['min_sigma']
		alpha = selection_dict['min_alpha']

		#store better results:
		value = np.zeros((nf,mu))
		for i in range(mu):
			value[:,i] = eggholder(min_x[j][0,i],min_x[j][1,i],min_x[j][2,i],min_x[j][3,i],min_x[j][4,i])
		min_f.append(value)
		eps = abs(obj-value[0,:])

		#Add error
		EPS[j] = np.average(eps)

		#Add counter
		j = j+1



	print 'done'
	return {'min_x':min_x, 'min_f':min_f, 'off':off, 'EPS': EPS, 'j': j}


