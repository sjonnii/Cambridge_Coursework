import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
import math
import scipy

def plot_data_internal (X, y):
	x_min , x_max = X[ : , 0 ].min() - .5 , X[ : , 0 ].max() + .5
	y_min , y_max = X[ : , 1 ].min() - .5 , X[ : , 1 ].max() + .5
	xx , yy = np.meshgrid (np.linspace (x_min , x_max , 100) , \
	np.linspace(y_min , y_max , 100))
	plt.figure()
	plt.xlim(xx.min() , xx.max())
	plt.ylim(yy.min() , yy.max())
	ax = plt.gca ()
	ax.plot (X[y == 0 , 0] , X[y == 0 , 1] , 'ro ', label = 'Class 1')
	ax.plot (X[y == 1 , 0] , X[y == 1 , 1] , 'bo ', label = 'Class 2')
	plt.xlabel ('X1 ')
	plt.ylabel ('X2 ')
	plt.title ('Plot data')
	plt.legend (loc = 'upper left', scatterpoints = 1 , numpoints = 1)
	return xx , yy

def plot_data (X, y):
	xx , yy = plot_data_internal(X, y)
	plt.show ()

def plot_ll (ll, ll2):
	plt.figure()
	ax =plt.gca()
	plt.xlim(0 , len(ll) + 2)
	plt.ylim(min(ll2) - 0.1 , max(ll) + 0.1)
	ax.plot(np. arange (1 , len(ll) + 1) , ll , 'r-')
	ax.plot(np. arange (1 , len(ll2) + 1) , ll2 , 'b-')
	plt.xlabel('Steps ')
	plt.ylabel('Average log - likelihood ')
	plt.title('Plot Average Log - likelihood Curve ')
	plt.legend(['Training Data', 'Test Data'], loc='best')
	plt.show()

def sigmoid(z):
	s = 1/(1+np.exp(-z))
	return s

def initialize_beta(dim):
    b = np.zeros((dim))
    
    return b

def likelihood (b,X,Y):
	A = sigmoid(np.dot(X,b))
	likelihood = np.sum(Y*np.log(A)+(1-Y)*np.log(np.exp(-np.dot(X,b))*A))
	likelihood = likelihood/Y.shape[0]
	return likelihood

def likelihood_prior(b,X,Y,sigma):
	A = sigmoid(np.dot(X,b))
	dim = Y.shape[0]
	likeli = np.sum(Y*np.log(A)+(1-Y)*np.log(np.exp(-np.dot(X,b))*A))
	likeli = likeli - (1/2) *  dim * (np.log((2*np.pi)) + np.log(sigma)) - 1/2 * np.dot(b.T,b)*(1/sigma)
	#likeli = likeli.reshape((likeli.shape[0]))
	return likeli


def gradient (b, X, Y):
	A = sigmoid(np.dot(X,b))
	db = np.dot((Y-A).T, X).T
	return db

def gradient_ascent(b,X,Y, iterations, learning_rate, test_x, test_y):
	likelihoods_train = []
	likelihoods_test = []

	for i in range(iterations):
		loglike_train = likelihood(b,X,Y)
		loglike_test = likelihood(b,test_x, test_y)
		db = gradient(b,X,Y)
		b = b + learning_rate*db

		likelihoods_train.append(loglike_train)
		likelihoods_test.append(loglike_test)

	return b, likelihoods_train, likelihoods_test

def gradient_ascent_Gprior(b,X,Y, iterations, learning_rate, sigma):
	likelihoods = []

	for i in range(iterations):
		loglike = likelihood_prior(b,X,Y,sigma)
		db = gradient(b,X,Y) - b
		b = b + learning_rate*db
		likelihoods.append(loglike)

	return b, likelihoods


def prediction(b,X, threshold):

	prediction = np.zeros((1,X.shape[0]))
	probabilities = sigmoid(np.dot(X,b))
	for i in range(probabilities.shape[1]):
		prediction = probabilities > threshold
	prediction = prediction.astype(int)
	return prediction, probabilities

def add_bias(X):
	ones = np.ones((X.shape[0],1))
	X_bias = np.c_[ones, X]
	return X_bias

def plot_predictive_distribution(X, y, b, threshold):
	xx , yy = plot_data_internal(X, y)
	ax = plt.gca()
	X_predict = np.concatenate((xx.ravel().reshape((-1,1)), \
	yy.ravel().reshape((-1,1))),1)
	X_predict = add_bias(X_predict)
	_,Z = prediction(b, X_predict, threshold)
	Z = Z.reshape(xx.shape)
	cs2 = ax.contour(xx , yy , Z, cmap = 'RdBu', linewidths = 2)
	plt.clabel(cs2 , fmt = '%2.1f', colors = 'k', fontsize = 14)
	plt.show()

def plot_predictive_distribution_expand(X, y, b, threshold, l, train_set):
	xx , yy = plot_data_internal(X, y)
	ax = plt.gca()
	X_predict = np.concatenate((xx.ravel().reshape((-1,1)), \
	yy.ravel().reshape((-1,1))),1)
	X_predict = expand_inputs(l, X_predict, train_set)
	X_predict = add_bias(X_predict)
	_,Z = prediction(b, X_predict, threshold)
	Z = Z.reshape(xx.shape)
	cs2 = ax.contour(xx , yy , Z, cmap = 'RdBu', linewidths = 2)
	plt.clabel(cs2 , fmt = '%2.1f', colors = 'k', fontsize = 14)
	plt.show()

def plot_roc (targets, probabilities):
	fpr, tpr, thresholds = roc_curve(targets, probabilities)
	roc_auc = auc(fpr, tpr)

	plt.title('ROC Curve')
	plt.plot(fpr, tpr, 'b')
	label = ('AUC = %0.2f' % roc_auc)
	plt.legend(loc='lower right')
	plt.plot([0,1],[0,1],'r--')
	plt.xlim([-0.1,1.2])
	plt.ylim([-0.1,1.2])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show()

	return roc_auc

def expand_inputs (l, X, Z):
	X2 = np.sum(X**2 , 1)
	Z2 = np.sum(Z**2 , 1)
	ones_Z = np.ones(Z.shape[ 0 ])
	ones_X = np.ones(X.shape[ 0 ])
	r2 = np.outer(X2 , ones_Z ) - 2 * np.dot(X, Z.T) + np.outer(ones_X , Z2)
	return np.exp(-0.5/l**2*r2)

def negative_loglike(b,X,Y):
	# A = sigmoid(np.dot(X,b))
	# likeli = np.sum(Y*np.log(A)+(1-Y)*np.log(np.exp(-np.dot(X,b))*A))
	# likeli = -(likeli - (1/2) * np.dot(b.T,b))
	likeli = likelihood_prior(b,X,Y,1)
	likeli = -likeli
	return likeli


def negative_gradient(b,X,Y):
	A = sigmoid(np.dot(X,b))
	db = -np.dot((Y-A).T, X).T
	return db


def optimize_scipy(X,Y):
	b_init = initialize_beta((X.shape[1],))
	b_map, likelihood, _ = scipy.optimize.fmin_l_bfgs_b(negative_loglike, b_init, negative_gradient, args = (X,Y))
	return b_map, likelihood

def hessian(X, b_map,sigma):
	elements = sigmoid(np.dot(X, b_map)) * (1-sigmoid(np.dot(X, b_map)))
	elements = np.diag(elements)
	hessian = np.dot(np.dot(X.T, elements), X)
	hessian = hessian + np.identity(801)*(1/sigma)
	return hessian

def predictive_distribution(X, b_map, chol):
	mean = np.dot(b_map,X.T)
	chol_inv = np.linalg.inv(chol)
	chol_dot = np.dot(chol_inv.T,chol_inv)
	diags = np.dot(np.dot(X, chol_dot), X.T)
	diag_mat = np.diag(diags)
	K = ((1+(np.pi*diag_mat/8)))**(-1/2)
	vec = K*mean
	probabilities = sigmoid(vec)
	return probabilities

def prediction_predictive(X, bmap, chol):
	predictions = []
	probabilities = predictive_distribution(X, bmap, chol)
	predictions = probabilities > 0.5
	predictions = predictions.astype(int)
	return predictions, probabilities

def plot_predictive_distribution_expand_(X, y, b, train_set, chol,l):
	xx , yy = plot_data_internal(X, y)
	ax = plt.gca()
	X_predict = np.concatenate((xx.ravel().reshape((-1,1)), \
	yy.ravel().reshape((-1,1))),1)
	X_predict = expand_inputs(l, X_predict, train_set)
	X_predict = add_bias(X_predict)
	_,Z = prediction_predictive(X_predict, b, chol)
	Z = Z.reshape(xx.shape)
	cs2 = ax.contour(xx , yy , Z, cmap = 'RdBu', linewidths = 2)
	plt.clabel(cs2 , fmt = '%2.1f', colors = 'k', fontsize = 14)
	plt.show()

def model_evidence (hess, bmap, X, Y, sigma):
	sign, logdet = np.linalg.slogdet(hess)
	dim = train_Y.shape[0]
	m_evidence = likelihood_prior(bmap,X, Y, sigma) + (dim/2)*np.log(2*np.pi) -(1/2)*logdet
	return m_evidence

#MAIN CODE
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

#Create data
X_ = np.loadtxt('X.txt')
y = np.loadtxt('y.txt')
train_X, test_X, train_Y, test_Y = train_test_split(X_,y, test_size = 0.2, random_state = 3)
train_x = add_bias(train_X)
test_x = add_bias(test_X)
train_y = train_Y.reshape(train_Y.shape[0],1)
test_y = test_Y.reshape(test_Y.shape[0],1)
#plot the data and the class it belongs to
plot_data(X_,y)

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

#train using our training data

b = initialize_beta((3,1))
b, likelihoods_train, likelihoods_test = gradient_ascent(b,train_x,train_y,100,0.001, test_x, test_y)
print (likelihoods_train[99])
print (likelihoods_test[99])

#Make prediction on our test set
test_y_predict, probabilities_test_y = prediction(b, test_x, 0.5)

#print the confusion matrix and then plot likelihood per iteration of optimization

plot_ll(likelihoods_train, likelihoods_test)
print (confusion_matrix(test_y,test_y_predict))
plot_predictive_distribution(X_, y, b, 0.5)

#plot ROC and calculate area under curve
auc = plot_roc(test_y, probabilities_test_y)
print (auc)

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

#expand our inputs through a set of RBFs
l = [0.01, 0.1, 1]

data_expand = {}

for j in range(len(l)):
	data_expand['train_expand{0}'.format(j)] = expand_inputs(l[j], train_X, train_X)
	data_expand['test_expand{0}'.format(j)] = expand_inputs(l[j], test_X, train_X)
	data_expand['train_expand{0}'.format(j)] = add_bias(data_expand['train_expand{0}'.format(j)]) #do this in the logistic function
	data_expand['test_expand{0}'.format(j)] = add_bias(data_expand['test_expand{0}'.format(j)])
# #(800,801) shape

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#make predictions etc

b1 = initialize_beta((801,1))
b2 = initialize_beta((801,1))
b3 = initialize_beta((801,1))

b1, ex_likelihoods_train1,ex_likelihoods_test1 = gradient_ascent(b1, data_expand['train_expand0'],train_y, 2000, 0.01, data_expand['test_expand0'], test_y)
b2, ex_likelihoods_train2,ex_likelihoods_test2 = gradient_ascent(b2, data_expand['train_expand1'],train_y, 2000, 0.001, data_expand['test_expand1'], test_y)
b3, ex_likelihoods_train3,ex_likelihoods_test3 = gradient_ascent(b3, data_expand['train_expand2'],train_y, 2000, 0.0001, data_expand['test_expand2'], test_y)

# print(ex_likelihoods_train1[])
# print(ex_likelihoods_test1[1999])
# print(ex_likelihoods_train2[1999])
# print(ex_likelihoods_test2[1999])
# print(ex_likelihoods_train3[1999])
# print(ex_likelihoods_test3[1999])

#Make predictions and plot predictive distribution -> Gerdu thetta ad falli Sjonni
predict_expand1, probabilities_expand1 = prediction(b1, data_expand['test_expand0'], 0.5)
print (confusion_matrix(test_y, predict_expand1))
plot_predictive_distribution_expand(X_, y, b1, 0.5, 0.01, train_X)
auc1 = plot_roc(test_y, probabilities_expand1)
print (auc1)

predict_expand2, probabilities_expand2 = prediction(b2, data_expand['test_expand1'], 0.5)
print (confusion_matrix(test_y, predict_expand2))
plot_predictive_distribution_expand(X_, y, b2, 0.5, 0.1, train_X)
auc2 = plot_roc(test_y, probabilities_expand2)
print (auc2)

predict_expand3, probabilities_expand3 = prediction(b3, data_expand['test_expand2'], 0.5)
print (confusion_matrix(test_y, predict_expand3))
plot_predictive_distribution_expand(X_, y, b3, 0.5, 1, train_X)
auc3 = plot_roc(test_y, probabilities_expand3)
print (auc3)

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------


# #We multiply our likelihood with a Guassian prior to calculate MAP. This changes the gradient a bit, it gets -B
# By doing this we are normalizing by penalising to large Betas


b1_prior = initialize_beta((801,1))
b2_prior = initialize_beta((801,1))
b3_prior = initialize_beta((801,1))

b1_prior, likelihoods1_prior = gradient_ascent_Gprior(b1_prior, data_expand['train_expand0'],train_y, 3000, 0.01,1)
b2_prior, likelihoods2_prior = gradient_ascent_Gprior(b2_prior, data_expand['train_expand1'],train_y, 3000, 0.001,1)
b3_prior, likelihoods3_prior = gradient_ascent_Gprior(b3_prior, data_expand['train_expand2'],train_y, 3000, 0.0001,1)

print(likelihood_prior(b1_prior,data_expand['train_expand0'],train_y,1))
print(likelihood_prior(b2_prior,data_expand['train_expand1'],train_y,1))
print(likelihood_prior(b3_prior,data_expand['train_expand2'],train_y,1))
print(likelihood_prior(b1_prior,data_expand['test_expand0'],test_y,1))
print(likelihood_prior(b2_prior,data_expand['test_expand1'],test_y,1))
print(likelihood_prior(b3_prior,data_expand['test_expand2'],test_y,1))


predict_expand1_prior, probabilities_expand1_prior = prediction(b1_prior, data_expand['test_expand0'], 0.5)
print (confusion_matrix(test_y, predict_expand1_prior))
#plot_predictive_distribution_expand(X_, y, b1_prior, 0.5, 0.01, train_X)
auc1_prior = plot_roc(test_y, probabilities_expand1_prior)
print (auc1_prior)

predict_expand2_prior, probabilities_expand2_prior = prediction(b2_prior, data_expand['test_expand1'], 0.5)
print (confusion_matrix(test_y, predict_expand2_prior))
plot_predictive_distribution_expand(X_, y, b2_prior, 0.5, 0.1, train_X)
auc2_prior = plot_roc(test_y, probabilities_expand2_prior)
print (auc2_prior)

predict_expand3_prior, probabilities_expand3_prior = prediction(b3_prior, data_expand['test_expand2'], 0.5)
print (confusion_matrix(test_y, predict_expand3_prior))
#plot_predictive_distribution_expand(X_, y, b3_prior, 0.5, 1, train_X)
auc3_prior = plot_roc(test_y, probabilities_expand3_prior)
print (auc3_prior)

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#ADDITIONAL EXERSICES


bmap, likelihoodss = optimize_scipy(data_expand['train_expand1'], train_Y)
hess = hessian(data_expand['train_expand2'], bmap,1) 
chol = np.linalg.cholesky(hess)
preds, probs = prediction_predictive(data_expand['test_expand1'], bmap, chol)
print (confusion_matrix(test_y, preds))
#plot_predictive_distribution_expand_(X_,y, bmap,train_X, chol,0.1)
print(negative_loglike(bmap, data_expand['train_expand1'], train_Y))
print(negative_loglike(bmap, data_expand['test_expand1'], test_Y))



#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#define the hyper parameters for the grid search
sigma_ = [0.1,0.25,0.5,0.75,1]
l_ = [0.1, 0.25, 0.5, 0.75, 1]
evidence = []
#grid search
for i in range(len(l_)):
	print ('loop')
	data_ev = expand_inputs(l_[i], train_X, train_X)
	data_ev = add_bias(data_ev)
	for j in range(len(sigma_)):
		bmap = optimize_scipy(data_ev, train_Y)
		hess = hessian(data_ev, bmap, sigma_[j])
		eve = model_evidence(hess, bmap, data_ev, train_Y, sigma_[j])
		evidence.append(eve)
print (evidence)


gogn = expand_inputs(0.5, train_X, train_X)
gogn_test = expand_inputs(0.5, test_X, train_X)
gogn = add_bias(gogn)
gogn_test = add_bias(gogn_test)
bmap_best, likeli = optimize_scipy(gogn, train_Y)
print (gogn.shape)
print (bmap_best.shape)
hessi = hessian(gogn, bmap_best,1)
choli = np.linalg.cholesky(hessi)
preds_best, probs_best = prediction_predictive(gogn_test, bmap_best, choli)
print (confusion_matrix(test_y, preds_best))
plot_predictive_distribution_expand_(X_,y, bmap_best,train_X, choli,0.5)
print(negative_loglike(bmap_best, gogn, train_Y))
print(negative_loglike(bmap_best, data_expand['test_expand1'], test_Y))




