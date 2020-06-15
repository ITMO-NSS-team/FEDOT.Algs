from scipy.optimize import differential_evolution, dual_annealing
from copy import deepcopy
import numpy as np
from AllTokensParamsOptimizer import *



class TokensBank:

	def __init__(self, tokens_to_bank):
		self.tokens_to_bank = tokens_to_bank
		self.bank = []

	@staticmethod
	def fitness_wrapper_for_de(params, *args):
		individ, t, target_TS, chromo_idx = args
		individ.chromo[chromo_idx].put_params(params)
		return individ.fitness(t, target_TS)

	def make_freqs(self, t, target_TS, freqs_bounds):
		wmin, wmax = freqs_bounds
		freqs, y = FrequencyProcessor.fft(t, target_TS, wmin, wmax, c=10)
		freqs, y = FrequencyProcessor.findmaxfreqs(freqs, y)
		freqs, y = FrequencyProcessor.keyw(freqs, y, crit=0.8)
		return freqs

	def optimize_key_freqs(self, w, t, target_TS, popsize, full_imp_opt=False):
		for token in self.tokens_to_bank:
			if token.type != 'Periodic':
				continue
			token = token.copy()
			bounds = deepcopy(token.bounds)
			eps = 0.005
			for freq in w:
				for ii in range(1):
					bounds[1] = (freq*(1-eps), freq*(1+eps))
					res = differential_evolution(self.fitness_wrapper_for_de, bounds, args=(Individ([token]), t, target_TS, 0),
												 popsize=popsize, workers=1)
					token.put_params(res.x)
					if type(token) in imp_relations.keys() and full_imp_opt:
						tmp_ind = Individ([ImpComplex(token.copy())])
						opt = ComplexImpTokenParamsOptimizer(tmp_ind, 0, optimizer='global')
						opt.optimize_params(t, target_TS)
						self.bank.append(tmp_ind.chromo[0])
					else:
						self.bank.append(token.copy())

	def optimize_trend(self, t, target_TS):
		tokens = []
		for token in self.tokens_to_bank:
			if token.type == 'NonPeriodic':
				tokens.append(token)
		if len(tokens) != 0:
			ind = Individ(tokens)
			opt = NonPeriodicAllTokensParamsOptimizer(ind)
			opt.optimize_params(t, target_TS)
			self.bank.extend(deepcopy(tokens))
			x = target_TS-ind.value(t)
			return x
		return target_TS				

	def make_bank(self, t, target_TS, freqs_bounds, popsize=5, full_imp_opt=False):
		target_TS = self.optimize_trend(t, target_TS)
		w = self.make_freqs(t, target_TS, freqs_bounds)
		self.optimize_key_freqs(w, t, target_TS, popsize, full_imp_opt)
		return self.bank