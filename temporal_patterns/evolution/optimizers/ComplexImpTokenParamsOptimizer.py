from Individ import *

class ComplexImpTokenParamsOptimizer:

	def __init__(self, individ, chromo_idx, optimizer='global'):
		self.individ = individ
		self.chromo_idx = chromo_idx
		self.token = self.individ.chromo[chromo_idx]
		self.optimizer = optimizer
		if self.token.type != 'ImpComplex':
			raise
		# self.token_individ = Individ(chromo=self.token.imps, used_value='Plus', used_fitness=self.individ.fitness)

	@staticmethod
	def fitness_wrapper(params, *args):
		individ, t, target_TS, chromo_idx = args
		individ.chromo[chromo_idx].put_params(params)
		return individ.fitness(t, target_TS)

	def optimize_params_local(self, t, target_TS):
		for chromo_idx in range(len(self.token_individ.chromo)):
			gen = self.token_individ.chromo[chromo_idx]
			params0 = []
			if gen.type == 'NonPeriodic':
				params0.extend(gen.params)
			# if params0:
				res = minimize(self.fitness_wrapper, params0, args=(self.token_individ, t, target_TS, chromo_idx), method='Nelder-Mead')
				gen.put_params(res.x)

	def optimize_params_global(self, t, target_TS):
		T = (1/self.token.pattern.params[1])#*self.token.pattern.params[2]
		for chromo_idx in range(len(self.token_individ.chromo)):
			bound = []
			gen = self.token_individ.chromo[chromo_idx]

			im1 = np.sum(self.token_individ.chromo[chromo_idx - 1].params[1:]) if chromo_idx > 0 else t[0]
			ip1 = self.token_individ.chromo[chromo_idx + 1].params[1] if chromo_idx < len(self.token_individ.chromo) - 2 else t[-1]

			if gen.type == 'NonPeriodic':
				bound.extend(gen.params)
				bound1 = []
				for j in range(len(bound)):
					if j == 0:
						# a = 0.5
						# bound1.append([bound[j]*(1-a), bound[j]*(1+a)])
						if bound[j] >= 0:
							# bound1.append([0, 1])
							bound1.append([bound[j]*0.9, 1])
						else:
							# bound1.append([-1, 0])
							bound1.append([-1, bound[j]*0.9])
					elif j == 1:
						# a = 0.1
						# bound1.append([bound[j]*(1-a), bound[j]*(1+a)])
						# bound1.append([im1+0.5*T, ip1-0.5*T])
						bound1.append([bound[j]-0.25*T, bound[j]+0.25*T])
					elif j in (2, 3):
						a = 0.1
						bound1.append([0, bound[j]*(1+a)])
						# bound1.append([0, bound[j]*(1+a)])
					else:
						a = 0.1
						bound1.append([bound[j]*(1-a), bound[j]*(1+a)])
						# bound1.append([0, bound[j]*(1+a)])
						# bound1.append([0, 3])
				bound = bound1
				for b in bound:
					if b[0] > b[1]:
						b[0], b[1] = b[1], b[0]
				# print(bound, 'bound!')
				popsize = 50
				# res = differential_evolution(self.fitness_wrapper, bound, args=(self.token_individ, t, target_TS, chromo_idx), popsize=popsize)
				res = dual_annealing(self.fitness_wrapper, bound, args=(self.token_individ, t, target_TS, chromo_idx), maxiter=popsize)
				gen.put_params(res.x)

			
	def optimize_params(self, t, target_TS):
		self.token.init_imps(t)
		self.token_individ = Individ(chromo=self.token.imps, used_value='Plus', used_fitness=self.individ.fitness)
		if self.optimizer == 'global':
			self.optimize_params_global(t, target_TS)
		else:
			self.optimize_params_local(t, target_TS)
		self.token.fix = True
