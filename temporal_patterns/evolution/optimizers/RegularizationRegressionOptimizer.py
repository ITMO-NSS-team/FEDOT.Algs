from sklearn.linear_model import Lasso, LinearRegression
from Individ import *


class RegularizationRegressionOptimizer:

	def __init__(self, individ):
		self.individ = individ

	def make_amplitudes_eq_1(self):
		for i in self.individ.chromo:
			if i.type == 'Periodic':
				i.params[0] = 1
			if i.type == 'ImpComplex':
				i.make_ampls_eq_1(1)
			# if i.type == "Product":
			# 	for j in i.multipliers:
			# 		j.params[0] = 1

	def linear_regression(self, t, target_TS):
		self.make_amplitudes_eq_1()
		periodic_chromo = self.individ.get_chromo(["Periodic", "ImpComplex"])
		if not periodic_chromo:
			return
		features = np.transpose(np.array(list(map(lambda x: x.value(t), periodic_chromo))))
		model = LinearRegression()
		model.fit(features, target_TS)
		for i in range(len(periodic_chromo)):
			if periodic_chromo[i].type == 'Product':
				periodic_chromo[i].multipliers[0].params[0] = model.coef_[i]
			elif periodic_chromo[i].type == 'ImpComplex':
				periodic_chromo[i].make_ampls_eq_1(model.coef_[i])
			else:
				periodic_chromo[i].params[0] = model.coef_[i]

	def lasso(self, t, target_TS, alpha=0.001):
		if not self.individ.opt_by_lasso:
			self.make_amplitudes_eq_1()
			periodic_chromo = self.individ.get_chromo(["Periodic", "ImpComplex"])
			trend_chromo = self.individ.get_chromo(["NonPeriodic"])
			trendval = np.zeros(len(t))
			if trend_chromo:
				trend = Individ(trend_chromo)
				trendval = trend.value(t)
			if not periodic_chromo:
				return
			features = np.transpose(np.array(list(map(lambda x: x.value(t), periodic_chromo))))
			model = Lasso(alpha)
			model.fit(features, target_TS-trendval)
			for i in range(len(periodic_chromo)):
				if periodic_chromo[i].type == 'Product':
					periodic_chromo[i].multipliers[0].params[0] = model.coef_[i]
				elif periodic_chromo[i].type == 'ImpComplex':
					periodic_chromo[i].make_ampls_eq_1(model.coef_[i])
				else:
					periodic_chromo[i].params[0] = model.coef_[i]
			self.individ.del_low_functions()
			self.linear_regression(t, target_TS)
			self.individ.opt_by_lasso = True

