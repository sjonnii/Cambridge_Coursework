## Tabu Search algorithm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm

def eggholder(x):
	summa = 0
	xin = [x[0],x[1],x[2],x[3],x[4]]
	for i in range(len(xin)-1):
		val = (-(xin[i+1] + 47)*np.sin(np.sqrt(np.abs(xin[i+1]+(xin[i]/2)+47))) - xin[i] *np.sin(np.sqrt(np.abs(xin[i] - (xin[i+1]+47)))))
		summa += val
	return summa


#Generate a starting point randomly
base = np.random.uniform(-512,512,5)

#set hyperparameters and initialize necessary lists
delta = 300
threshold = 1e-50

STM = []
STM_size = 7
STM.append(base)

MTM = []
MTM_size = 4
MTM.append(base)
MTM_fvals = []
MTM_fvals.append(eggholder(base))

LTM = []
LTM.append(base)

counter = 0
SI_threshold = 13
SD_threshold = 20
SR_threshold = 23

#bases1 and 2 are used for 2D plotting
# bases1 = []
# bases2 = []


#Compute Local Search
for j in range(10000):
	#increase and decrease each control variable by delta and evaluate function at each point and append to lists.
	func_values = []
	control_variables = []
	for i in range(5): 
		tmp_inc = np.copy(base) #make copies of x0 
		tmp_dec = np.copy(base)
		#Make sure to keep the constraints
		if tmp_inc[i] + delta < 512:
			tmp_inc[i] = tmp_inc[i] + delta
		else:
			tmp_inc[i] = 512
		func_values.append(eggholder(tmp_inc))
		control_variables.append(tmp_inc)
		#Make sure to keep the contraints
		if tmp_dec[i] - delta > -512:
			tmp_dec[i] = tmp_dec[i] - delta
		else:
			tmp_dec[i] = -512 
		func_values.append(eggholder(tmp_dec))
		control_variables.append(tmp_dec)

	#Find the best allowed (non-tabu) move
	boolean = True
	while boolean:
		min_fval = min(func_values)
		min_fval_index = func_values.index(min_fval)
		min_control_var = control_variables[min_fval_index]

		#calc euclidiean distance for that point to every point in STM --> Not used here but thought it could be useful for the algorithm
		#dist = []
		#for i in range (len(STM)):
			#dist.append(np.linalg.norm(min_control_var-STM[i]))

		#Check if the point is in STM. If it is I remove it and look agian, if it's not then bool = True and the While loop breaks
		if any((min_control_var == x).all() for x in STM):
			func_values.pop(min_fval_index)
			control_variables.pop(min_fval_index)
		else:
			boolean = False

	#Do pattern move
	vector = min_control_var - base
	pattern_var = min_control_var + vector
	pattern_move = eggholder(pattern_var)
	#First see if pattern move will violate constraints
	if any(min_control_var + vector > 512) or any(min_control_var + vector > -512):
		#skip pattern move if it will violate constraints
		base = min_control_var
		if len(STM) < STM_size:
			STM.append(base)
		else:
			STM.pop(0)
			STM.append(base)
	else:
		#if it does not violate constraints, check if it improves the object function. If it does, we do it
		if pattern_move < min_fval:
			base = pattern_var
			if len(STM) < STM_size:
				STM.append(base)
			else:
				STM.pop(0)
				STM.append(base)
		#if it doesnt improve, we don't do it
		else:
			base = min_control_var
			if len(STM) < STM_size:
				STM.append(base)
			else:
				STM.pop(0)
				STM.append(base)


	#see if the solution is better than what we have in MTM. If it is, we input it to the MTM and rest counter, otherwise we add to the counter
	if (eggholder(base) < max(MTM_fvals) and not any((base == x).all() for x in MTM)) or len(MTM) < MTM_size:
		if len(MTM) < MTM_size:
			MTM.append(base)
			MTM_fvals.append(eggholder(base))
		else:
			MTMindex = MTM_fvals.index(max(MTM_fvals))
			MTM_fvals.pop(MTMindex)
			MTM_fvals.append(eggholder(base))
			MTM.pop(MTMindex)
			MTM.append(base)
			counter = 0
	else:
		counter += 1

	#Search intensification. If counter gets to the threshold, we intensifie the serach
	if counter == SI_threshold:
		base = np.multiply((1.0/4),(MTM[0]+MTM[1]+MTM[2]+MTM[3]))
	else:
		pass

	#Search diversification Keep it simple, if counter goes to threshold, we choose a new random location to start searching in
	if counter == SD_threshold:
		base = np.random.uniform(-512,512,5)

	#If counter goes to threshold, we halve the step size
	if counter == SR_threshold:
		if delta < 0.00000000001:
			delta = delta
		else:
			delta = delta/2.0
		newbase_index = MTM_fvals.index(min(MTM_fvals))
		base = MTM[newbase_index]
		counter = 0



#print the lowest value obtained during the search
print '---'
print 'min:'
print min(MTM_fvals)


#Contour plot for the 2D
# xmin, xmax, xstep = -512, 512, 0.1
# ymin, ymax, ystep = -512, 512, 0.1
# startpoint1 = bases1[0]
# startpoint2 = bases2[0]
# endpoint1 = bases1[j]
# endpoint2 = bases2[j]

# x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
# xlisti = [x,y]
# z = eggholder(xlisti)
# cp = plt.contourf(x,y,z)
# plt.colorbar(cp)
# plt.plot(bases1,bases2,'g*')
# plt.plot(startpoint1,startpoint2,'r+', label = 'size 30')
# plt.plot(endpoint1,endpoint2,'b+',label = 'size 30')
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.show()




	









