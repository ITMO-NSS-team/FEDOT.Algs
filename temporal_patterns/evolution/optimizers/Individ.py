from Tokens import *
from scipy.optimize import differential_evolution
from scipy.optimize import minimize, leastsq, differential_evolution, shgo, dual_annealing, basinhopping, brute
from sklearn.linear_model import Lasso, LinearRegression
import time as tm
import sys
from time import perf_counter





class Individ:

	def __init__(self, chromo=None, fit=None, opt_by_lasso=False, kind='init', tokens=None, used_fitness=None, used_value='Plus', \
				use_fix_funcs_in_val=True):
		if chromo == None:
			chromo = []
		self.chromo = chromo
		self.fit = fit
		self.opt_by_lasso = opt_by_lasso
		self.kind = kind
		self.fix = False
		if tokens == None:
			tokens = Function.__subclasses__()
		self.tokens = tokens
		self.used_value = used_value
		self.used_fitness = used_fitness
		self.use_fix_funcs_in_val = use_fix_funcs_in_val

	def copy(self):
		ind = deepcopy(self)
		ind.opt_by_lasso = False
		self.fix = False
		return ind
		
	def init_formula(self, max_len = 2):
		len_formula = np.random.randint(1, max_len+1)
		for i in range(len_formula):
			add_function = random.choice(self.tokens).copy()
			self.chromo.append(add_function)

	def formula(self, with_params=False):
		if self.used_value == "Plus":
			joinWith = '+'
		else:
			joinWith = '*'
		return joinWith.join(list(map(lambda x: x.name(with_params), self.chromo)))

	def show(self, t, target_TS, with_patterns=False):
		plt.figure(self.formula(True))
		if with_patterns:
			plt.subplot(2,1,1)
		plt.plot(t, target_TS, label='target', color='black')
		plt.plot(t, self.value(t), label='model', color='orange')
		# plt.title(self.formula(True), fontdict={'fontsize': 10})
		plt.grid(True)
		plt.xlabel('t')
		plt.ylabel('time series')
		if with_patterns:
			plt.subplot(2,1,2)
			mn = 0
			for idx in range(len(self.chromo)):
				gen = self.chromo[idx]
				if idx > 0:
					mn = self.chromo[idx-1].params[0]
				plt.plot(t, gen.value(t) + 2*mn, label=gen.name(True))
			plt.title('patterns')
			plt.xlabel('t')
			plt.ylabel('pattern')
			plt.grid(True)
			plt.legend()
		plt.legend()

	def value(self, t):
		if self.use_fix_funcs_in_val:
			temp = self.chromo
		else:
			temp = list(filter(lambda x: x.fix == False, self.chromo))

		if self.used_value == 'Plus':
			return reduce(lambda val, x: val + x, list(map(lambda x: x.value(t), temp)))
		elif self.used_value == 'Product':
			return reduce(lambda val, x: val * x, list(map(lambda x: x.value(t), temp)))
		else:
			sys.exit()

	def fitness(self, *args):
		t, target_TS = args
		if self.used_fitness == None:
			self.fit = np.linalg.norm(self.value(t) - target_TS)/np.linalg.norm(target_TS)
		else:
			self.fit = self.used_fitness(t, target_TS)
		return self.fit

	def get_params(self, _type=None):
		params = []
		for i in self.chromo:
			if not i.fix:
				if _type == None or i.type == _type:
					params.extend(list(i.get_params()))
		return np.array(params)

	def get_params1(self, _type=None):
		chromo = list(filter(lambda x: ((x.type==_type or _type==None) and not x.fix), self.chromo))
		params = []
		for i in chromo:
			params.append(list(i.get_params()))
		return np.array(params)

	def put_params1(self, params, _type=None):
		params = list(params)
		for i in self.chromo:
			if not i.fix:
				if _type == None or i.type == _type:
					params_for_function = []
					for j in range(i.number_params):
						if params:
							params_for_function.append(params.pop(0))
						else:
							params_for_function.append(0)
					i.put_params(params_for_function)
					if not params:
						break

	def put_params(self, params, *_type):
		if len(_type)==0:
			_type == None
		chromo = list(filter(lambda x: ((x.type in _type or _type==None) and not x.fix), self.chromo))
		start = 0
		for gen in chromo:
			gen.put_params(params[start: start + gen.number_params])
			start += gen.number_params

	def get_chromo(self, types=None):
		if types == None:
			return self.chromo
		return list(filter(lambda x: x.type in types, self.chromo))


	def del_functions_with_special_properties(self, t, target_TS):
		for i in self.chromo:
			if i.name() == 'Gauss':
				if (i.params[0] < 0 or i.params[2] > np.max(t) or i.params[2] < np.min(t) or i.params[1]**2 > 9*np.var(target_TS)) and len(self.chromo) > 1:
					self.chromo.remove(i)
			elif i.name() == 'Power':
				if (i.params[0] < 0 or i.params[1] < 1 or abs(i.params[1]) > 2.2) and len(self.chromo) > 1:
					self.chromo.remove(i)

	def del_nonuniq_functions(self):
		for i in range(len(self.chromo)):
			for j in range(i+1, len(self.chromo)):
				if self.chromo[i] == self.chromo[j]:
					self.chromo.remove(self.chromo[j])

	def del_low_functions(self):
		self.del_nonuniq_functions()
		chromo = self.get_chromo(['Periodic', 'Product', 'ImpComplex'])
		if len(chromo) == 0: return
		crit = np.max(list(map(lambda x: x.params[0], chromo)))
		for i in chromo:
			if abs(i.params[0]) < 0.1*crit and len(chromo) > 1:
				# print("Delete1", i.name(True))
				self.chromo.remove(i)

	def del_near_period_functions(self, t, target_TS):
		i = 0
		while i < len(self.chromo):
			j = i + 1
			while j < len(self.chromo):
				if self.chromo[i].type == 'Periodic' and self.chromo[j].type == 'Periodic'\
					and self.chromo[i].number_params > 1:
					if abs(self.chromo[i].params[1] - self.chromo[j].params[1])/np.max([self.chromo[i].params[1], self.chromo[j].params[1]]) < 0.05:
						# if self.chromo[i].params[0] < self.chromo[j].params[0]:
						if np.linalg.norm(self.chromo[i].value(t)-target_TS) > np.linalg.norm(self.chromo[j].value(t)-target_TS):
							self.chromo[i], self.chromo[j] = self.chromo[j], self.chromo[i]
						self.chromo.pop(j)
						self.chromo[i].fix = False
				j += 1
			i += 1
		pass	

	def imp_to_complex_imp(self, t, target_TS, stop_idx=None):
		if stop_idx == None or stop_idx > len(self.chromo)-1 or stop_idx < 0:
			stop_idx = len(self.chromo)-1
		for idx in range(stop_idx+1):
			func = self.chromo[idx]
			if type(func) in imp_relations.keys():
				self.chromo[idx] = ImpComplex(func)

	def mutation(self, increase_prob=0.5, mut_intensive=2):
		ind = self.copy()
		ind.kind = 'mutation'
		mut_intensive = random.randint(1, mut_intensive)
		add_functions = random.choices(self.tokens, k=mut_intensive)
		for idx in range(len(add_functions)):
			add_functions[idx] = add_functions[idx].copy()
		if np.random.uniform() < increase_prob:
			ind.chromo.extend(add_functions)
		else:
			for idx in range(len(ind.chromo)):
				if add_functions:
					ind.chromo[idx] = add_functions.pop(random.randint(0, len(add_functions)-1))
				else:
					break
			if add_functions:
				ind.chromo.extend(add_functions)
		return [ind]

	def crossover(self, ind2, increase_prob=0.5, cross_intensive=2):
		ind1 = self.copy()
		ind2 = ind2.copy()

		ind1.kind = 'crossover'
		ind2.kind = 'crossover'

		cross_intensive = random.randint(1, np.min([cross_intensive, len(ind1.chromo), len(ind2.chromo)]))
		add_idxs1 = np.random.choice(len(ind1.chromo), size=cross_intensive, replace=False)
		add_idxs2 = np.random.choice(len(ind2.chromo), size=cross_intensive, replace=False)
		if np.random.uniform() < increase_prob:
			for idx in add_idxs1:
				gen = ind1.chromo[idx].copy()
				gen.fix = False
				ind2.chromo.append(gen)
			for idx in add_idxs2:
				gen = ind2.chromo[idx].copy()
				gen.fix = False
				ind1.chromo.append(gen)
		else:
			for idx1, idx2 in np.transpose([add_idxs1, add_idxs2]):
				ind1.chromo[idx1], ind2.chromo[idx2] = ind2.chromo[idx2], ind1.chromo[idx1]

		return [ind1, ind2]

	def product_mutation(self):
		tokens = list(filter(lambda x: x.type == 'Periodic' and x.name() != 'Sin', self.tokens))
		if len(tokens) == 0:
			return []
		ind = self.copy()
		idx = random.randint(0, len(ind.chromo)-1)
		if ind.chromo[idx].type == 'Product':
			ind.chromo[idx].multipliers.append(random.choice(tokens))
			ind.chromo[idx].fix = False
		else:
			ind.chromo[idx] = Product(multipliers=[ind.chromo[idx], random.choice(tokens)], fix=False)
		return [ind]
