import random
import numpy as np
from copy import deepcopy
from scipy import signal
from functools import reduce
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod, abstractproperty



class Token(ABC):
	@abstractmethod
	def value(self, t):
		pass

	@abstractmethod
	def get_params(self):
		pass

	@abstractmethod
	def put_params(self, params):
		pass

	def name(self, with_params=False):
		try:
			if with_params:
				return type(self).__name__ + str(list(self.params))
			return type(self).__name__
		except:
			return type(self).__name__

	def copy(self):
		return deepcopy(self)



class Function(Token):
	number_params = 0

	def __init__(self, params=None, bounds=None, fix=False):
		if bounds == None:
			self.bounds = [(0, 1) for i in range(self.number_params)]
		else:
			self.bounds = bounds
		if params == None:
			self.params = np.zeros(self.number_params)
			self.init_params()
		else:
			self.params = np.array(params, dtype=float)
		self.fix = fix

	def get_params(self):
		return self.params.copy()

	def put_params(self, params):
		self.params = np.array(params, dtype=float)

	def init_params(self):
		for i in range(len(self.params)):
			self.params[i] = np.random.uniform(self.bounds[i][0], self.bounds[i][1])

	def show(self, t, with_params=False):
		plt.figure(type(self).__name__)
		plt.plot(t, self.value(t))
		plt.title(self.name(with_params))
		plt.legend()
		plt.grid(True)
		plt.xlabel('t, s')
		plt.ylabel('Time Series, o.e.')
		plt.show()






# Trend Functions

class Power(Function):
	number_params = 3

	def __init__(self, params=None, bounds=[(-1, 1), (0.1, 3), (-1, 1)], fix=False):
		super().__init__(params=params, bounds=bounds, fix=fix)
		self.type = "NonPeriodic"

	def value(self, t):
		return self.params[0]*t**self.params[1] + self.params[2]




# Smooth Periodic Functions

class Sin(Function):
	number_params = 3

	def __init__(self, params=None, bounds=[(0, 1), (0.1, 6), (0, 1)], fix=False):
		super().__init__(params=params, bounds=bounds, fix=fix)
		self.type = "Periodic"

	def value(self, t):
		A = self.params[0]
		w = self.params[1]
		fi = self.params[2]
		return A*np.sin(2*np.pi*(w*t + fi))



# Pulses


class ImpSingle(Function):

	number_params = 6

	def __init__(self, params=None, bounds=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 3), (0, 3)], fix=False):
		super().__init__(params=params, bounds=bounds, fix=fix)
		self.type = "NonPeriodic"

	def value(self, t):
		A = self.params[0]
		T1 = self.params[1]
		T2 = self.params[2]
		T3 = self.params[3]
		p1 = self.params[4]
		p2 = self.params[5]

		cond1 = (t >= T1) & (t < T1 + T2)
		cond2 = (t >= T1 + T2) & (t <= (T1+T2+T3))
		m = np.zeros(len(t))
		if T2 != 0:
			m[cond1] = (abs(t[cond1] - T1)/T2)**abs(p1)
		if T3 != 0:
			m[cond2] = (abs(t[cond2] - (T1+T2+T3))/T3)**abs(p2)
		return A*m


class Imp(Function):

	number_params = 7

	def __init__(self, params=None, bounds=[(-1, 1), (0.04, 10), (0.05, 0.98), (0.05, 0.98), (0, 3), (0, 3), (0, 1)], fix=False):
		super().__init__(params=params, bounds=bounds, fix=fix)
		self.type = "Periodic"



	def value(self, t):
		A = self.params[0]
		fi = self.params[-1]
		T = 1/self.params[1]
		n1 = self.params[2]
		T1 = n1*T
		n2 = self.params[3]
		T2 = n2*(T - T1)
		T3 = T - T1 - T2

		p1 = self.params[4]
		p2 = self.params[5]

		t1 = (t + fi*T) % T

		cond1 = (t1 >= T1) & (t1 < T1 + T2)
		cond2 = (t1 >= T1 + T2) & (t1 <= T)
		m = np.zeros(len(t))
		if T2 != 0:
			m[cond1] = (abs(t1[cond1] - T1)/T2)**abs(p1)
		if T3 != 0:
			m[cond2] = (abs(t1[cond2] - T)/T3)**abs(p2)
		return A*m


# тип соответствия и сколько параметров после второго использовать
imp_relations = {
	Imp: [ImpSingle, 3],
}




class ImpComplex(Token):

	def __init__(self, pattern, fix=False):
		self.pattern = pattern.copy()
		self.number_params = self.pattern.number_params
		self.params = self.pattern.params
		self.fix = fix
		self.type = 'ImpComplex'
		self.imps = []
		self.sample_imps = [self.pattern]
		self.value_type = 'norm'

	def name(self, with_params=False):
		return 'Complex' + self.pattern.name(with_params)

	def name1(self, with_params=False):
		s = ''
		for i in self.imps:
			s += i.name(with_params)
		if with_params:
			return type(self).__name__ + str(self.pattern.params) + s
		return type(self).__name__ + '_' + self.pattern.name()

	def get_params(self):
		return self.pattern.get_params()

	def put_params(self, params):
		self.pattern.put_params(params)

	def value(self, t):
		if self.value_type == 'norm':
			self.init_imps(t)
			return self.value_imps(t)
		else:
			return self.sample_value(t)

	def init_imps(self, t):
		if len(self.imps) != 0:
			return
		nb_of_time_params = imp_relations[type(self.pattern)][1]
		T = list(map(lambda x: x, self.pattern.params[1: 1 + nb_of_time_params]))
		T[0] = 1/T[0]
		T1 = deepcopy(T)
		for idx in range(len(T)-1):
			T[idx] = T1[idx+1]*(T1[0] - np.sum(T[:idx]))
		T[-1] = T1[0] - np.sum(T[:-1])
		add = np.sum(T)
		sm = -self.pattern.params[-1]*add + T[0]
		mxt = t[-1]
		while sm < mxt:
			new_params = [self.pattern.params[0], sm]
			for i in range(nb_of_time_params - 1):
				new_params.append(T[1 + i])
			new_params.extend(self.pattern.params[nb_of_time_params+1: -1])
			new_imp = imp_relations[type(self.pattern)][0](new_params)
			self.imps.append(new_imp)
			sm += add

	def value_imps(self, t):
		return reduce(lambda val, x: val + x, list(map(lambda x: x.value(t), self.imps)))

	def init_params(self):
		self.pattern.init_params()

	def get_noise_sample(self, t, a=0.02):
		self.init_imps(t)
		for i in self.imps:
			for j in range(len(i.params)):
				i.params[j] *= np.random.uniform(1 - a, 1 + a)
		return self.value_imps(t)

	def del_imps_out_of_range_in_comlex_imp(self, t):
		mxt = t[-1]
		for imp in self.imps:
			if imp.params[1] > mxt:
				self.imps.remove(imp)

	def del_imps_in_area_of_zero_value(self, t, value):
		for imp in self.imps:
			impPeriod = (imp.params[1], np.sum(imp.params[1:]))
			if (value[(t >= impPeriod[0]) & (t <= impPeriod[1])] == 0).all():
				self.imps.remove(imp)
				print(imp.name(True), 'was deleted because out of', impPeriod)


	def make_ampls_eq_1(self, pA=1):
		A = list(map(lambda x: x.params[0], self.imps))
		if len(A) != 0:
			mxA = np.max(np.abs(A))
			if mxA == 0:
				mxA = 1
		for i in self.imps:
			i.params[0] *= pA/mxA
		self.params[0] = pA

	def make_properties(self, t):
		mas = []
		for i in range(len(self.imps[0].params)):
			ps = list(map(lambda x: x.params[i], self.imps))
			if i == 1:
				tmp = []
				tmp.append(ps[0])
				tmp.extend(list(map(lambda x,y: y-x, ps[:-1], ps[1:])))
				ps = tmp
			mas.append(ps)
		return mas
			
	def make_sample(self, t, probs):
		mxt = t[-1]
		self.sample_imps = []
		imp = self.imps[0].copy()
		sm = 0
		while sm < mxt:
			temp_imp = imp.copy()
			m = np.mean(probs[1])
			sigm = np.var(probs[1])**0.5
			sm += np.random.normal(m, sigm)
			temp_imp.params[1] = sm
			for i in range(len(temp_imp.params)):
				if i != 1:
					m = np.mean(probs[i])
					sigm = np.var(probs[i])**0.5
					temp_imp.params[i] = np.random.normal(m, sigm)
			self.sample_imps.append(temp_imp)

	def make_full_sample(self, t):
		probs = self.make_properties(t)
		self.make_sample(t, probs)

	def sample_value(self, t):
		return reduce(lambda val, x: val + x, list(map(lambda x: x.value(t), self.sample_imps)))


class Product(Token):

	def __init__(self, multipliers=[], fix=False):
		self.multipliers = []
		self.multipliers.extend(deepcopy(multipliers))
		if len(self.multipliers) == 0:
			self.init_multipliers()
			for i in self.multipliers:
				i.init_params()
		self.fix = fix
		self.type = 'Product'
		self.params = np.array([1])

	def value(self, t):
		self.params[0] = reduce(lambda val, x: val * x, list(map(lambda x: x.params[0], self.multipliers)))
		return reduce(lambda val, x: val * x, list(map(lambda x: x.value(t), self.multipliers)))

	def get_params(self):
		params = np.array([])
		for i in self.multipliers:
			params = np.append(params, i.get_params())
		return params

	def put_params(self, params):
		params = list(params)
		for i in self.multipliers:
			if not i.fix:
				params_for_function = []
				for j in range(i.number_params):
					if params:
						params_for_function.append(params.pop(0))
					else:
						params_for_function.append(0)
				i.set_params(params_for_function)
				if not params:
					break

	def name(self, with_params=False):
		s = '('
		for i in self.multipliers:
			s += i.name(with_params)
		s += ')'
		if with_params:
			return type(self).__name__ + s
		return type(self).__name__

	def init_multipliers(self, n=2, samples=[Sin, Imp]):
		samples = random.choices(samples, k=n)
		for i in range(len(samples)):
			samples[i] = samples[i]()
		self.multipliers.extend(samples)
