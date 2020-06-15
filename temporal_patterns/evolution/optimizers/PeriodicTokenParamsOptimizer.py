from scipy.optimize import differential_evolution, dual_annealing
from FrequencyProcessor import *
from copy import deepcopy


class PeriodicTokenParamsOptimizer:

	def __init__(self, individ, chromo_idx, optimizer='differential_evolution'):
		self.individ = individ
		self.chromo_idx = chromo_idx
		self.optimizer = optimizer
		self.token = self.individ.chromo[chromo_idx]
		if self.token.type != "Periodic":
			raise

	@staticmethod
	def fitness_wrapper(params, *args):
		self, t, target_TS = args
		self.token.put_params(params)
		return self.individ.fitness(t, target_TS)	

	def optimize_params(self, t, target_TS, fix_each_function=True):
		if not self.token.fix:
			x = target_TS - self.individ.value(t) + self.token.value(t)
			wmin, wmax = self.token.bounds[1][0], self.token.bounds[1][1]
			freq = FrequencyProcessor.find_freq_for_summand(t, x, wmin, wmax)
			eps = 0.005
			bounds = deepcopy(self.token.bounds)
			bounds[1] = (freq*(1-eps), freq*(1+eps))
			popsize = 5
			res = differential_evolution(self.fitness_wrapper, bounds, args=(self, t, target_TS), popsize=popsize)
			self.token.put_params(res.x)
			self.token.fix = fix_each_function


		
