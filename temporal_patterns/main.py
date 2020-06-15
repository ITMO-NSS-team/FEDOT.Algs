# from Population import *
from evolution.Population import *



# sample individ for the formation of target Time Series
ind = Individ([Sin([0.2, 0.08, 0]), Imp([1, 0.01, 0.9 ,0.5, 0.4, 0, 0]), Power([0.001, 1, 0])])

# the time must be consistent with the frequency search range 
t = np.arange(0, 2000, 1)
# target Time Series must be centered and normalized
target_TS = ind.value(t)

# tokens that the expression will be collected from
tokens = [Sin(), Imp(), Power()]

population_size = 20

# the regularization coefficient for lasso
regularization_coef = 0.1

# boundaries of the required frequencies of periodic components
freqs_bounds = [0.005, 0.1]

# number of generations in genetic algorithm
generations = 100

# Bank will be updated once in a given number of generations
gen_to_bank_update = 25

# use over-optimization of pulses (if an uneven period and amplitude is expected)
full_imp_opt = False

# The pop size parameter for differential evolution for optimizing periodic components
DE_popsize = 10


pop = Population(tokens=tokens, population_size=population_size, alpha=regularization_coef)
pop.GA(t, target_TS, freqs_bounds=freqs_bounds, generations=generations, gen_to_bank_update=gen_to_bank_update, full_imp_opt=full_imp_opt, popsize=DE_popsize)

best = pop.population[0]

print('expression: ', best.formula(True))
print('best fit: ', best.fit)

# show results
best.show(t, target_TS)


plt.show()