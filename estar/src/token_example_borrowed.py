# if __name__ == '__main__':
# 	print('Это была 2 часть', flush=True)
# else:
# 	# print('Это была 2 part', flush=True)
# 	pass

import random
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from scipy.optimize import differential_evolution
from scipy import signal
from sklearn.linear_model import Lasso
import time as tm
from copy import deepcopy


t = np.arange(0.1, 300, 0.1)

class Function(object):
	"""docstring for Function"""
	def __init__(self, arg):
		self.arg = arg

	def copy(self):
		return deepcopy(self)

	def name(self, with_params=False):
		if with_params:
			return type(self).__name__ + str(self.params)
		return type(self).__name__

	def get_params(self):
		return self.params.copy()

	def put_params(self, params):
		self.params = np.array(params)

	def show(self, t):
		plt.figure(type(self).__name__)
		plt.plot(t, self(t))
		plt.show()

	def __call__(self, t):
		return self.value(t)

class PeriodicFunction(Function):
	def __init__(self):
		super().__init__()


class TrendFunction(Function):
	def __init__(self):
		super().__init__()




class Sin(PeriodicFunction):

	def __init__(self, params=None, fix=False):
		self.number_params = 3
		if params == None:
			params = [0. for i in range(self.number_params)]
		self.params = np.array(params)
		self.fix = fix

	def value(self, t):
		return self.params[0]*np.sin(self.params[1]*t + 2*np.pi*self.params[2])

class Vector(PeriodicFunction):

	def __init__(self, params=None, fix=False):
		self.number_params = 1
		if params == None:
			params = [0. for i in range(self.number_params)]
		self.params = np.array(params)
		self.fix = fix

	def value(self, t):
		return self.params[0]*np.sin(0.1*t)


# class RImp(PeriodicFunction):

# 	def __init__(self, params=None, fix=False):
# 		self.number_params = 3
# 		if params == None:
# 			params = [0. for i in range(self.number_params)]
# 		self.params = np.array(params)
# 		self.fix = fix

# 	def value(self, t):
# 		return self.params[0]*signal.square(self.params[1]*t + 2*np.pi*self.params[2])



# class RImp(PeriodicFunction):

# 	def __init__(self, params=None, fix=False):
# 		self.number_params = 4
# 		if params == None:
# 			params = [0. for i in range(self.number_params)]
# 		self.params = np.array(params)
# 		self.fix = fix

# 	def value(self, t):
# 		A = self.params[0]
# 		T1 = self.params[1]
# 		T2 = self.params[2]
# 		fi = self.params[3]

# 		m = np.zeros(len(t))
# 		cond = ((t+fi*(T1+T2)) % (T1+T2) < T1)
# 		m[cond] = 1
# 		m[~cond] = -1
# 		return A*m


class Power(PeriodicFunction):
	def __init__(self, params=None, fix=False):
		self.number_params = 2
		if params == None:
			params = [0. for i in range(self.number_params)]
		self.params = np.array(params)
		self.fix = fix

	def value(self, t):
		return self.params[0]*t**self.params[1]# + self.params[2]

# class Poly3(TrendFunction):
# 	def __init__(self, params=None, fix=False):
# 		self.number_params = 4
# 		if params == None:
# 			params = [0. for i in range(self.number_params)]
# 		self.params = np.array(params)
# 		self.fix = fix

# 	def value(self, t):
# 		return self.params[0]*t**3 + self.params[1]*t**2 + self.params[2]*t + self.params[3]




class Individ:

	def __init__(self, chromo=None, fit=None, opt_by_lasso=False, kind='init'):
		if chromo == None:
			chromo = []
		self.chromo = chromo
		self.fit = fit
		self.opt_by_lasso = opt_by_lasso
		self.kind = kind
		self.fix = False

	def copy(self):
		ind = deepcopy(self)
		ind.opt_by_lasso = False
		self.fix = False
		return ind
		
	def init_formula(self, max_len = 2, function_class=PeriodicFunction):
		len_formula = np.random.randint(1, max_len+1)
		for i in range(len_formula):
			add_function = random.choice(function_class.__subclasses__())()
			self.chromo.append(add_function)

	def value(self, t):
		return reduce(lambda val, x: val + x, list(map(lambda x: x.value(t), self.chromo)))

	def formula(self, with_params=False):
		return '+'.join(list(map(lambda x: x.name(with_params), self.chromo)))

	def show(self, t, target_TS):
		plt.figure(type(self).__name__)
		plt.plot(t, target_TS, label='original')
		plt.plot(t, self(t), label='approximation')
		plt.legend()
		plt.grid(True)
		plt.show()

	def sum_of_abs_amplitudes(self):
		return reduce(lambda x,y: x+y, list(map(lambda x: abs(x.params[0]), self.chromo)))

	def fitness(self, *args):
		t, target_TS = args
		self.fit = np.linalg.norm(self.value(t) - target_TS)/np.linalg.norm(target_TS)# + 0.75*self.sum_of_abs_amplitudes()
		return self.fit

	def get_params(self):
		params = []
		for i in self.chromo:
			if not i.fix:
				params.extend(list(i.get_params()))
		return np.array(params)

	def put_params(self, params):
		params = list(params)
		for i in self.chromo:
			if not i.fix:
				params_for_function = []
				for j in range(i.number_params):
					if params:
						params_for_function.append(params.pop(0))
					else:
						params_for_function.append(0)
				i.put_params(params_for_function)
				if not params:
					break

	def fitness_wrapper_for_optimize(params, *args):
		self, t, target_TS = args
		self.put_params(params)
		return self.fitness(t, target_TS)

	def fitness_wrapper_for_optimize_step_by_step(params, *args):
		self, t, target_TS, chromo_idx = args
		self.chromo[chromo_idx].put_params(params)
		return self.fitness(t, target_TS)

	def del_low_functions(self):
		for i in self.chromo:
			if abs(i.params[0]) < 0.0001 and len(self.chromo) > 1:
				self.chromo.remove(i)

	def del_near_period_functions(self):
		i = 0
		while i < len(self.chromo):
			j = i + 1
			while j < len(self.chromo):
				if self.chromo[i].name() == self.chromo[j].name() and self.chromo[i].number_params > 1 and self.chromo[j].number_params > 1:
					if abs(self.chromo[i].params[1] - self.chromo[j].params[1])/max(self.chromo[i].params[1], self.chromo[j].params[1]) < 0.2:
						if self.chromo[i].params[1] < self.chromo[j].params[1]:
							self.chromo[i], self.chromo[j] = self.chromo[j], self.chromo[i]
						self.chromo.pop(j)
						self.chromo[i].fix = False
				j += 1
			i += 1

	def optimize_params(self, t, target_TS):
		if not self.fix:
			bound = []
			for i in self.chromo:
				if (i.name() == 'Sin' or i.name() == 'RImp') and not i.fix:
					bound.extend([(0, 1), (0, 10), (0, 1)])
				elif(i.name() == 'Power') and not i.fix:
					bound.extend([(-1, 1), (-2, 2)])

			print(bound)
			res = differential_evolution(Individ.fitness_wrapper_for_optimize, bound, args=(self, t, target_TS))
			self.put_params(res.x)
			self.del_low_functions()
			self.del_near_period_functions()

			self.fix = True
			return res

	# def optimize_params_step_by_step(self, t, target_TS, from_beginning=False):
	# 	if from_beginning:	
	# 		for i in self.chromo:
	# 			for j in range(len(i.params)):
	# 				i.params[j] = 0.
	# 	for chromo_idx in range(len(self.chromo)):
	# 		bound = []
	# 		if self.chromo[chromo_idx].name() == 'Sin':
	# 			bound.extend([(0, 1), (0, 10), (0, 2*np.pi)])
	# 		res = differential_evolution(Individ.fitness_wrapper_for_optimize_step_by_step, bound, args=(self, t, target_TS, chromo_idx))
	# 		self.chromo[chromo_idx].put_params(res.x)
	# 	# self.del_low_functions()
	# 	# self.sort_by_amplitude()

	# def make_fix(self):


	def optimize_params_step_by_step(self, t, target_TS, from_beginning=False, fix_each_function=True):
		if from_beginning:	
			for i in self.chromo:
				if not i.fix:
					for j in range(len(i.params)):
						i.params[j] = 0.
		for chromo_idx in range(len(self.chromo)):
			if not self.chromo[chromo_idx].fix:
				bound = []
				if self.chromo[chromo_idx].name() == 'Sin':
					bound.extend([(0, 1), (2*np.pi/abs(t[-1] - t[0]), 0.2*2*np.pi/abs(t[0]-t[1])), (0, 1)])
				elif self.chromo[chromo_idx].name() == 'RImp':
					bound.extend([(-1, 1), (10*abs(t[0]-t[1]) ,abs(t[-1] - t[0])), (10*abs(t[0]-t[1]) ,abs(t[-1] - t[0])), (0, 1)])
				elif self.chromo[chromo_idx].name() == 'Power':
					bound.extend([(-1, 1), (-2, 2)])
				elif self.chromo[chromo_idx].name() == 'Poly3':
					bound.extend([(-1, 1), (-1, 1), (-1, 1), (-1, 1)])
				elif self.chromo[chromo_idx].name() == 'Vector':
					bound.extend([(-1, 1)])
				


				res = differential_evolution(Individ.fitness_wrapper_for_optimize_step_by_step, bound, args=(self, t, target_TS, chromo_idx))
				self.chromo[chromo_idx].put_params(res.x)
				self.chromo[chromo_idx].fix = fix_each_function
		self.del_low_functions()
		self.del_near_period_functions()

	def lasso(self, t, target_TS, alpha=0.01):
		# flag = reduce(lambda x, y: x*y, list(map(lambda x: x.fix, self.chromo)))
		if not self.opt_by_lasso:
			self.make_amplitudes_eq_1()
			features = np.transpose(np.array(list(map(lambda x: x.value(t), self.chromo))))
			model = Lasso(alpha)
			model.fit(features, target_TS)
			# print(model.coef_)
			for i in range(len(self.chromo)):
				self.chromo[i].params[0] = model.coef_[i]

			# coef = np.var(target_TS_test)/np.var(self.value(t))

			self.del_low_functions()
			self.opt_by_lasso = True

			# coef = np.var(target_TS_test)/np.var(self.value(t))
			# coef = 1
			# for i in range(len(self.chromo)):
			# 	self.chromo[i].params[0] *= coef

	def sort_by_amplitude(self):
		idx = np.argsort(list(map(lambda x: x.params[0], self.chromo)))
		self.chromo = [self.chromo[i] for i in idx[::-1]]

	def make_amplitudes_eq_1(self):
		for i in self.chromo:
			i.params[0] = 1

	def mutation(self, function_class=PeriodicFunction):
		ind = self.copy()
		ind.kind = 'mutation'
		add_function = random.choice(function_class.__subclasses__())()
		if np.random.uniform() < 0.5:
			ind.chromo.append(add_function)
		else:
			ind.chromo[np.random.randint(0, len(ind.chromo))] = add_function
		return [ind]

	# def crossover(self, ind2):
	# 	ind1 = self.copy()
	# 	ind2 = ind2.copy()

	# 	if np.random.uniform() < 0.2:
	# 		gen1 = random.choice(ind1.chromo)
	# 		gen2 = random.choice(ind2.chromo)

	# 		ind1.chromo.append(gen2.copy())
	# 		ind2.chromo.append(gen1.copy())
	# 	else:
	# 		if len(ind1.chromo) < len(ind2.chromo):
	# 			ind1, ind2 = ind2, ind1
	# 		for i in range(len(ind1.chromo)):
	# 			if np.random.uniform() < 0.5:
	# 				if i >= len(ind2.chromo):
	# 					ind2.chromo.append(ind1.chromo[i].copy())
	# 				else:
	# 					ind1.chromo[i], ind2.chromo[i] = ind2.chromo[i], ind1.chromo[i]
	# 	return [ind1, ind2]

	def crossover(self, ind2):
		ind1 = self.copy()
		ind2 = ind2.copy()

		ind1.kind = 'crossover'
		ind2.kind = 'crossover'

		if np.random.uniform() < 0.2:
			gen1 = random.choice(ind1.chromo)
			gen2 = random.choice(ind2.chromo)

			ind1.chromo.append(gen2.copy())
			ind2.chromo.append(gen1.copy())
		else:
			for i in range(min(len(ind1.chromo), len(ind2.chromo))):
				if np.random.uniform() < 0.5:
					idx1 = np.random.randint(0, len(ind1.chromo))
					idx2 = np.random.randint(0, len(ind2.chromo))
					ind1.chromo[idx1], ind2.chromo[idx2] = ind2.chromo[idx2], ind1.chromo[idx1]
		return [ind1, ind2]


	def __call__(self, t):
		return self.value(t)




class Population:

	def __init__(self, population=None, function_class=PeriodicFunction):
		if population == None:
			population = []
		self.population = population
		self.function_class = function_class

	def new_population(self, n=1):
		for i in range(n):
			new_ind = Individ()
			new_ind.init_formula(function_class=self.function_class)
			self.population.append(new_ind)

	def crossover(self):
		parents = random.sample(self.population, 2)
		childs = parents[0].crossover(parents[1])
		self.population.extend(childs)

	def mutation(self):
		mutants = []
		parents = random.sample(self.population, 3)
		for i in parents:
			mutants.extend(i.mutation(function_class=self.function_class))
		self.population.extend(mutants)

	def optimize_params(self, t, target_TS):
		for i in self.population:
			# i.optimize_params(t, target_TS)
			i.optimize_params_step_by_step(t, target_TS)
			i.lasso(t, target_TS, alpha=0.01)
			# i.sort_by_amplitude()

	def fitsort(self, t, target_TS):
		for i in self.population:
			i.fitness(t, target_TS)
		idx = np.argsort(list(map(lambda x: x.fit, self.population)))
		self.population = [self.population[i] for i in idx]

	def GA_step(self, t, target_TS):
		self.crossover()
		self.mutation()
		self.optimize_params(t, target_TS)
		self.fitsort(t, target_TS)
		try:
			self.population = self.population[:5]
		except:
			pass

	def GA(self, t, target_TS, generations = 1):
		self.fits = []
		self.new_population(5)
		self.optimize_params(t, target_TS)
		self.fitsort(t, target_TS)
		for i in range(generations):
			self.GA_step(t, target_TS)
			self.fits.append(self.population[0].fit)
			for j in self.population[:1]:
				print(j.formula(True))
				print(j.fit, j.kind)
				print(j)
			print("n_pops ------> {} on iteration {} \n".format(len(self.population), i))








ind1 = Individ([Sin([1, 1, 1]), Sin([0.9, 5, 1]), Sin([0.5, 2, 0.3]), Sin([0.7, 7, 2.8]), Sin([1, 1.5, 2]), Sin([1, 0.1, 0]),\
 Sin([1, 10, 1])])

# ind1 = Individ([Power([0.005, 1]), Power([0.004, 1.2]), Sin([1, 1, 1]), Sin([0.9, 5, 1]), Sin([0.5, 2, 0.3]), Sin([0.7, 7, 2.8]), Sin([1, 1.5, 2]), Sin([1, 0.1, 0]),\
#  Sin([1, 10, 1])])

# ind1 = Individ([Sin([1, 1, 1]), Sin([0.9, 5, 1]), Sin([0.5, 2, 0.3]), Sin([0.7, 7, 2.8]), Sin([1, 1.5, 2]), Sin([1, 0.1, 0]),\
#  Sin([1, 10, 1])])

# ind1 = Individ([RImp([0.5, 2, 1]), RImp([0.5, 0.5, 0]), RImp([0.5, 5, 2.5]), Sin([1, 1, 1]), Sin([0.5, 8, 0])], Sin([1, 4, 1]))
# ind1 = Individ([RImp([1, 1, 0]), Sin([1, 1, np.pi/2])])


# ind1 = Individ([Power([0.1, 1.2, 0]), Sin([10, 1, 1])])

formula = ind1.formula(True)

target_TS = ind1.value(t)# + np.random.normal(0, 0.5, len(t))
target_TS += 0.04*t
aaaa = np.max(abs(target_TS))
target_TS /= np.max(abs(target_TS))









# import pandas as pd


# a = pd.read_csv('1995_01_01_00-00--1995_06_01_00-00.csv')
# a = a[:-2]
# a = a.loc[:, "sossheig-0-time": 'sossheig-0-value']

# a1 = pd.read_csv('1995_06_01_01-00--1995_12_28_00-00.csv')
# a1 = a1[:-2]
# a1 = a1.loc[:, "sossheig-0-time": 'sossheig-0-value']

# a2 = pd.read_csv('1995_12_28_01-00--1996_06_01_00-00.csv')
# a2 = a2[:-2]
# a2 = a2.loc[:, "sossheig-0-time": 'sossheig-0-value']

# a3 = pd.read_csv('1996_01_06_01-00--1997_01_01_00-00.csv')
# a3 = a3[:-2]
# a3 = a3.loc[:, "sossheig-0-time": 'sossheig-0-value']

# a4 = pd.read_csv('1997_01_01_01-00--1997_06_01_00-00.csv')
# a4 = a4[:-2]
# a4 = a4.loc[:, "sossheig-0-time": 'sossheig-0-value']

# a5 = pd.read_csv('1997_06_01_01-00--1889_01_01_00-00.csv')
# a5 = a5[:-2]
# a5 = a5.loc[:, "sossheig-0-time": 'sossheig-0-value']

# a6 = pd.read_csv('1998_01_01_01-00--1998_01_06_00-00.csv')
# a6 = a6[:-2]
# a6 = a6.loc[:, "sossheig-0-time": 'sossheig-0-value']

# a7 = pd.read_csv('1998_06_01_01-00--1999_01_01_00-00.csv')
# a7 = a7[:-2]
# a7 = a7.loc[:, "sossheig-0-time": 'sossheig-0-value']

# a8 = pd.read_csv('1999_01_01_01-00--2000_01_01_00-00.csv')
# a8 = a8[:-2]
# a8 = a8.loc[:, "sossheig-0-time": 'sossheig-0-value']

# a9 = pd.read_csv('2000_01_01_01-00--01_01_2002_00-00.csv')
# a9 = a9[:-2]
# a9 = a9.loc[:, "sossheig-0-time": 'sossheig-0-value']



# a = a.append(a1, ignore_index=True)
# a = a.append(a2, ignore_index=True)
# a = a.append(a3, ignore_index=True)
# a = a.append(a4, ignore_index=True)
# a = a.append(a5, ignore_index=True)
# a = a.append(a6, ignore_index=True)
# a = a.append(a7, ignore_index=True)
# a = a.append(a8, ignore_index=True)
# a = a.append(a9, ignore_index=True)

# stop = int(1*len(a)) - 1
# # stop = 0
# # a = a[stop:]
# a = a[:stop]
# print(a)





# t_init = np.array(list(a['sossheig-0-time']))
# # t_init = (t_init - t_init[0])/(t_init[-1] - t_init[0])
# t_init = (t_init - t_init[0])/3600


# # t_init = np.linspace(0, 100, 1000)
# # a = RImp([1, 9, 3, 0])
# # target_TS_init = a.value(t_init)







# stop = int(0.6*len(t_init))
# # stop = len(t_init) - 1
# print(stop)

# t = t_init[:stop]

# t_test = t_init[stop:]

# target_TS_init = np.array(list(a['sossheig-0-value']))
# target_TS_init -= np.mean(target_TS_init)
# target_TS_init /= np.max(np.abs(target_TS_init))



# target_TS = target_TS_init[:stop]

# target_TS_test = target_TS_init[stop:]
# # print(t, y)

# print('dt={}, Wmax={}, Wmin={}'.format(t[1] - t[0], 0.1*2*np.pi/(t[1] - t[0]), 2*np.pi/(t[-1] - t[0])))

# plt.plot(t_init, target_TS_init)
# # # plt.plot(t, np.sin(0.1*2*np.pi/(t[1] - t[0]) * t))
# # # plt.plot(t[1:], target_TS[1:] - target_TS[:-1])
# plt.show()



print(0.2*2*np.pi/abs(t[0]-t[1]))
for i in range(1):
	pop = Population(function_class=PeriodicFunction)

	t1 = tm.perf_counter()
	pop.GA(t, target_TS, 50)
	t1 = tm.perf_counter() - t1


	print('time---->', t1)
	print('real form', formula)



	# print(pop.population[0].formula(True))
	# for i in pop.population[0].chromo:
	# 	print(i.fix)


	# t1 = tm.perf_counter()
	# pop.population[0].optimize_params_step_by_step(t, target_TS)
	# t1 = tm.perf_counter() - t1
	# print('time---->', t1)
	# print(pop.population[0].formula(True))



	pop.population[0].show(t, target_TS)
	# plt.figure(i)
	plt.plot(pop.fits)
	print('\n-----------END-------------------\n\n\n\n')
# plt.show()

# plt.figure('Dif')
# plt.plot(t, ind2.value(t)/aaaa)
# plt.plot(t, target_TS - pop.population[0].value(t))

# plt.show()


# plt.figure('Test')
# plt.plot(t_test, target_TS_test)
# plt.plot(t_test, pop.population[0].value(t_test))


t_init = t
target_TS_init = target_TS


plt.figure('All')
plt.plot(t_init, target_TS_init)
plt.plot(t_init, pop.population[0].value(t_init))
pop.population[0].sort_by_amplitude()
for i in pop.population[0].chromo:
	plt.plot(t_init, i.value(t_init))
plt.show()










# a = signal.square(1*t) * signal.square(1*t)
# ind = Individ([Sin([0.50590533, 1.00044389, 3.10738839]),RImp([4.43605429e-02, 5.00016814e+00, 1.72103566e-03]),RImp([6.50780854e-01, 9.99993701e-01, 7.77320476e-04])])
# a = ind.value(t)
# plt.plot(a)
# plt.show()




# pop.new_population(2)

# for i in pop.population:
# 	print(i.formula(True), i)
# print('---------------')

# pop.crossover()

# for i in pop.population:
# 	print(i.formula(True), i)
















# print(Function.__subclasses__())














"""
Мб примерно одинаковые по амплитуде составляющие, при оптимизации они могут быть упущены - в этом должна помогать мутация/кроссовер, ибо разные
индивиды могут быть оптимизированы по разному.
Оптимизация по шагам примерно в 5 раз быстрее, однако меньше точность, но |
Лассо обнуляет лишние элементы, избавляется от микродополнений и (возомжно) близких частотных составляющих - ПОКА НЕ ВИДНО
Лассо снижает значимость левых частот, от которых потом легче избавиться амплитудно, но близкие частоты не распознает
Близкие частоты уничтожаются до лассо!

ДЕ имеет тенденцию занулять составляющие в бесшумовых условиях, однако при искажениях достигаются минимумы с составляющими с малой амплитудой 
и "случайной" частотой
"""