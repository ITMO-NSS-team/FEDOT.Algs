from optimizers.Individ import *
from optimizers.AllTokensParamsOptimizer import *
from optimizers.RegularizationRegressionOptimizer import *
from optimizers.TokensBank import *

# from Individ import *
# from AllTokensParamsOptimizer import *
# from RegularizationRegressionOptimizer import *
# from TokensBank import *



class Population:

	def __init__(self, population=None, tokens=None, population_size=20, alpha=0.001):
		if population == None:
			population = []
		self.population = population
		if tokens == None:
			raise
		self.tokens = tokens
		self.population_size = population_size
		self.alpha = alpha

	def make_tokens(self, t, target_TS, freqs_bounds, popsize, full_imp_opt=False):
		bank = TokensBank(tokens_to_bank=self.tokens)
		self.tokens = bank.make_bank(t, target_TS, freqs_bounds, popsize, full_imp_opt)

	def new_population(self):
		for i in range(self.population_size):
			new_ind = Individ(tokens=self.tokens)
			new_ind.init_formula()
			self.population.append(new_ind)

	def crossover(self):
		parents = random.sample(self.population, len(self.population)//2)
		childs = parents[0].crossover(parents[1])
		self.population.extend(childs)

	def mutation(self):
		mutants = []
		parents = random.sample(self.population, len(self.population)//2)
		for i in parents:
			mutants.extend(i.mutation())
		self.population.extend(mutants)

	def optimize_params(self, t, target_TS):
		for ind in self.population:
			ind.del_near_period_functions(t, target_TS)
			opt = RegularizationRegressionOptimizer(ind)
			opt.lasso(t, target_TS, self.alpha)
			

	def fitsort(self, t, target_TS):
		for i in self.population:
			i.fitness(t, target_TS)
		idx = np.argsort(list(map(lambda x: x.fit, self.population)))
		self.population = [self.population[i] for i in idx]

	def make_hist(mas, normed=True, bins=100):
		h = np.histogram(mas, normed=normed, bins=bins)
		x = 0.5*(h[1][1:] + h[1][:-1])
		return [x, h[0]]

	def find_noise_distribution(self, t, target_TS):
		noise = target_TS - self.population[0].value(t)
		hist = Population.make_hist(noise, bins=int(len(noise)/50))
		pop_noise = Population(tokens=[Gauss])
		pop_noise.GA(hist[0], hist[1], 15)

		plt.figure('Noise distribution')
		plt.plot(hist[0], hist[1])
		plt.plot(hist[0], pop_noise.population[0].value(hist[0]))
		for i in pop_noise.population[0].chromo:
			plt.plot(hist[0] ,i.value(hist[0]))
		plt.show()

	def next_generation(self):
		fits = list(map(lambda x: x.fit, self.population))
		fits_sum = np.sum(fits)
		probabilities = list(map(lambda x: x/fits_sum, fits))
		for_remove = list(np.random.choice(self.population, size=len(self.population)-self.population_size, replace=False, p=probabilities))
		best = self.population[0]
		for ind in for_remove:
			self.population.remove(ind)
		if best not in self.population:
			self.population.append(best)


	def GA_step(self, t, target_TS):
		self.crossover()
		self.mutation()
		self.optimize_params(t, target_TS)
		self.fitsort(t, target_TS)
		self.fits.append(self.population[0].fit)
		

	def GA(self, t, target_TS, freqs_bounds, generations = 1, gen_to_bank_update=10, popsize=5, full_imp_opt=False):
		self.fits = []
		self.last_iteration = generations
		self.make_tokens(t, target_TS, freqs_bounds, popsize, full_imp_opt)
		self.new_population()
		self.optimize_params(t, target_TS)
		self.fitsort(t, target_TS)
		for i in range(1, generations):
			self.GA_step(t, target_TS)
			if i%gen_to_bank_update == 0:
				self.fitsort(t, target_TS)
				self.make_tokens(t, target_TS-self.population[0].value(t), freqs_bounds, popsize, full_imp_opt)
				# self.tokens.extend(self.population[0].chromo)
				for i in self.population:
					i.tokens = self.tokens
				print('iteration: ', i)
				print('num of tokens in bank ', len(self.tokens))
				print('fit: ', self.fits[-1])
			self.next_generation()
