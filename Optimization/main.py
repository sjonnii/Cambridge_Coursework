from evolution_strategy import eggholder
from evolution_strategy import initi
from evolution_strategy import recombination
from evolution_strategy import mutation
from evolution_strategy import selection
from evolution_strategy import evolution_strategy
import numpy as np

#define hyperparameters
n_x = 5
limits = np.array([[-512,512],[-512,512],[-512,512],[-512,512],[-512,512]])
obj    = -4000				# Objective Function
nf      = 1                 # length of the output vector 'f(x,y)'
mu      = 50                # parent population size
lambd  = 350                # offspring population size
gen     = 10000/mu          # number of generations

sel = '+'					#Selection method
rec_obj = 4				#control variables recombination
rec_str = 5				#Strategy parameter recombination

#Run the ES algorithm
ES_Dict = evolution_strategy(mu,lambd,gen,sel,rec_obj,rec_str,obj,nf,n_x,limits)
#results from the ES algorithm
min_f = ES_Dict['min_f'][100][0] #list
index_low = list(min_f).index(min(min_f))
lowest_f = min(min_f)
min_x = ES_Dict['min_x']
lowest_x = (min_x[100][:,index_low])

print lowest_f
print lowest_x
