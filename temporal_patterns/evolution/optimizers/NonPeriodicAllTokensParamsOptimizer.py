from scipy.optimize import minimize, differential_evolution
import numpy as np

class NonPeriodicAllTokensParamsOptimizer:

	def __init__(self, individ):
		self.individ = individ

	@staticmethod
	def fitness_wrapper(params, *args):
		individ, t, target_TS = args
		individ.put_params(params, 'NonPeriodic')
		return individ.fitness(t, target_TS)

	def optimize_params(self, t, target_TS):
		chromo = self.individ.get_chromo(['NonPeriodic'])
		if not chromo:
			return
		params0 = []
		for i in chromo:
			# params0.extend(i.params)
			params0.extend(i.bounds)
		# res = minimize(self.fitness_wrapper, params0, args=(self.individ, t, target_TS), method='Nelder-Mead')
		res = differential_evolution(self.fitness_wrapper, params0, args=(self.individ, t, target_TS),
									 popsize=5, workers=1)
		self.individ.put_params(res.x, 'NonPeriodic')
		# self.individ.del_functions_with_special_properties(t, target_TS)	


