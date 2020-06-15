from PeriodicTokenParamsOptimizer import *
from NonPeriodicAllTokensParamsOptimizer import *
from ComplexImpTokenParamsOptimizer import *
from Individ import *


class AllTokensParamsOptimizer:

	def __init__(self, individ):
		self.individ = individ

	def optimize_params(self, t, target_TS, ImpComplexOptimizer='global'):
		for chromo_idx in range(len(self.individ.chromo)):
			gen = self.individ.chromo[chromo_idx]
			if not gen.fix:
				if gen.type == 'Periodic':
					opt = PeriodicTokenParamsOptimizer(self.individ, chromo_idx)
					opt.optimize_params(t, target_TS)
					self.individ.imp_to_complex_imp(t, target_TS, stop_idx=chromo_idx)
					gen = self.individ.chromo[chromo_idx]
				if gen.type == 'Product':
					opt = ProductTokenParamsOptimizer(self.individ, chromo_idx)
					opt.optimize_params(t, target_TS)
				if gen.type == 'ImpComplex':
					opt = ComplexImpTokenParamsOptimizer(self.individ, chromo_idx, optimizer=ImpComplexOptimizer)
					opt.optimize_params(t, target_TS)
				if gen.type == 'NonPeriodic':
					opt = NonPeriodicAllTokensParamsOptimizer(self.individ)
					opt.optimize_params(t, target_TS)
				else:
					raise
				# opt.optimize_params(t, target_TS)
				gen.fix = True

		self.individ.del_low_functions()