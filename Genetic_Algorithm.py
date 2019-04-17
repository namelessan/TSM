import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


# 'Create class to handle "cities"'

# class Node:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

#     def distance(self, city):
#         xDis = abs(self.x - city.x)
#         yDis = abs(self.y - city.y)
#         distance = np.sqrt((xDis ** 2) + (yDis ** 2))
#         return distance

class Node:
    def __init__(self, name, lat, long):
        self.name = name
        self.lat = lat
        self.long = long

class Graph(ABC):
    def __init__(self, filename):
        self.cities = self.get_cities(filename)

    def get_cities(self, file_name):
        cities = []
        # read every line in the file that represent for each city,
        # return a list of cities object
        with open(file_name, 'r') as f:
            loop = True
            while loop:
                line = f.readline()
                if not line:
                    break
                info = line.split(',')
                city_name = info[0].strip()
                latitude = float(info[1].strip())
                longtitude = float(info[2].strip())
                city = Node(city_name, latitude, longtitude)
                cities.append(city)
        return cities

    def distance(self, des, start):
        # distance between 2 nodes
        return ((des.lat - start.lat)**2 + (des.long - start.long)**2)**0.5

    def cost(self, path):
        # total distance of the path
        total = 0.0
        for i in range(1, len(path)):
            total += self.distance(path[i], path[i-1])
        return total

    def fitness(self, path):
        # fitness of the path: the lower cost, the higher fitness
        return 1 / float(self.cost(path))

    @abstractmethod
    def find_shortest_path(self):
        pass
# 'Create a fitness function'

# class Fitness:
#     def __init__(self, path):
#         self.path = path
#         self.fit_distance = 0
#         self.fitness = 0.0

#     def path_distance(self):
#         if self.fit_distance == 0:
#             path_distance = 0
#             for i in range(len(self.path)):
#                 from_city = self.path[i]
#                 to_city = None
#                 if i + 1 < len(self.path):
#                     to_city = self.path[i + 1]
#                 else:
#                     to_city = self.path[0]
#                 path_distance += from_city.distance(to_city)
#             self.fit_distance = path_distance
#         return self.fit_distance

#     def path_fitness(self):
#         if self.fitness == 0:
#             self.fitness = 1 / float(self.path_distance())
#         return self.fitness

# 'Create our initial population'
# 'Route generator'

class Genetic(Graph):
    def __init__(self, filename):
        super().__init__(filename)

    def create_path(self, city_list):
        path = random.sample(city_list, len(city_list))
        return path

    # 'Create first "population" (list of paths)'


    def initial_populations(self, pop_size, city_list):
        populations = []

        for i in range(pop_size):
            populations.append(self.create_path(city_list))
        return populations

    # 'Create the genetic algorithm'
    # 'Rank individuals'


    def rank_paths(self, populations):
        fitness_results = {}
        for i in range(len(populations)):
            fitness_results[i] = self.fitness(populations[i])
        return sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=True)

    # 'Create a selection function that will be used to make the list of parent paths'


    def selection(self, pop_ranked, elite_size):
        selection_results = []
        df = pd.DataFrame(np.array(pop_ranked), columns=["Index", "Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()

        for i in range(elite_size):
            selection_results.append(pop_ranked[i][0])
        for _ in range(len(pop_ranked) - elite_size):
            pick = 100*random.random()
            for i in range(len(pop_ranked)):
                if pick <= df.iat[i, 3]:
                    selection_results.append(pop_ranked[i][0])
                    break
        return selection_results

    # 'Create mating pool'


    def mating_pool(self, populations, selection_results):
        mating_pool = []
        for i in range(len(selection_results)):
            index = selection_results[i]
            mating_pool.append(populations[index])
        return mating_pool

    # 'Create a crossover function for two parents to create one child'


    def breed(self, parent1, parent2):
        child = []
        child_p1 = []
        child_p2 = []

        geneA = int(random.random() * len(parent1))
        geneB = int(random.random() * len(parent1))

        start_gene = min(geneA, geneB)
        end_gene = max(geneA, geneB)

        for i in range(start_gene, end_gene):
            child_p1.append(parent1[i])

        child_p2 = [item for item in parent2 if item not in child_p1]

        child = child_p1 + child_p2
        return child

    # 'Create function to run crossover over full mating pool'


    def breed_population(self, mating_pool, elite_size):
        children = []
        length = len(mating_pool) - elite_size
        pool = random.sample(mating_pool, len(mating_pool))

        for i in range(elite_size):
            children.append(mating_pool[i])

        for i in range(length):
            child = self.breed(pool[i], pool[len(mating_pool)-i-1])
            children.append(child)
        return children

    # 'Create function to mutate a single path'


    def mutate(self, individual, mutation_rate):
        for swapped in range(len(individual)):
            if(random.random() < mutation_rate):
                swap_with = int(random.random() * len(individual))

                city1 = individual[swapped]
                city2 = individual[swap_with]

                individual[swapped] = city2
                individual[swap_with] = city1
        return individual

    # Create function to run mutation over entire population


    def mutate_population(self, populations, mutation_rate):
        mutated_pop = []

        for ind in range(len(populations)):
            mutated_ind = self.mutate(populations[ind], mutation_rate)
            mutated_pop.append(mutated_ind)
        return mutated_pop

    # Put all steps together to create the next generation


    def next_generation(self, current_gen, elite_size, mutation_rate):
        pop_ranked = self.rank_paths(current_gen)
        selection_results = self.selection(pop_ranked, elite_size)
        mating_pools = self.mating_pool(current_gen, selection_results)
        children = self.breed_population(mating_pools, elite_size)
        next_generation = self.mutate_population(children, mutation_rate)
        return next_generation

    # Final step: create the genetic algorithm


    def find_shortest_path(self, pop_size, elite_size, mutation_rate, generations):
        populations = self.initial_populations(pop_size, self.cities)
        print("Initial distance: " + str(1 / self.rank_paths(populations)[0][1]))

        for i in range(generations):
            populations = self.next_generation(populations, elite_size, mutation_rate)

        print("Final distance: " + str(1 / self.rank_paths(populations)[0][1]))
        best_path_index = self.rank_paths(populations)[0][0]
        best_path = populations[best_path_index]
        return best_path

# Running the genetic algorithm¶

GEN = Genetic('vietnam_cities.csv')
best = GEN.find_shortest_path(pop_size=100,
                  elite_size=20, mutation_rate=0.01, generations=500)
print([city.name for city in best])
# Create list of cities


# city_list = []

# for i in range(25):
#     city_list.append(Node(x=int(random.random() * 200),
#                           y=int(random.random() * 200)))

# Run the genetic algorithm
# find_shortest_path(populations=city_list, pop_size=100,
#                   elite_size=20, mutation_rate=0.01, generations=500)


# Plot the progress
# Note, this will win run a separate GA

# def find_shortest_pathPlot(populations, pop_size, elite_size, mutation_rate, generations):
#     pop = self.initial_populations(pop_size, populations)
#     progress = []
#     progress.append(1 / self.rank_paths(pop)[0][1])

#     for i in range(generations):
#         pop = self.next_generation(pop, elite_size, mutation_rate)
#         progress.append(1 / self.rank_paths(pop)[0][1])

#     plt.plot(progress)
#     plt.ylabel('Distance')
#     plt.xlabel('Generation')
#     plt.show()

# # Run the function with our assumptions to see how distance has improved in each generation

# find_shortest_pathPlot(populations=city_list, pop_size=100, elite_size=20, mutation_rate=0.01, generations=500)
