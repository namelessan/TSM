#!/usr/bin/env python3
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from sys import maxsize
from math import floor
from math import exp
import time
import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt


'''______________________Class object part___________________'''


class Node:
    def __init__(self, name=None, lat=0, long=0):
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
        # total += self.distance(path[0], path[-1])
        return total

    def fitness(self, path):
        # fitness of the path: the lower cost, the higher fitness
        return 1 / float(self.cost(path))

    def show_process(self, i, total):
        if i > 0 and total > 0:
            print('RUNNING: ', round(i/total*100, 2), '%')

    def find_nearest(self, cities, city):
        # find the nearest city of given city
        nearest = maxsize
        for city2 in cities:
            if city2 != city:
                distance = (city2.lat - city.lat)**2 + \
                    (city2.long - city.long)**2
                if distance < nearest:
                    nearest = distance
                    next_city = city2
        return next_city

    @abstractmethod
    def find_shortest_path(self):
        pass


class Nearest(Graph):
    def __init__(self, filename):
        super().__init__(filename)

    def find_shortest_path(self):
        path = []
        path_len = 0

        # initialize path and remain city
        path.append(self.cities[0])
        self.cities.pop(0)

        # find the path by the nearest city
        while self.cities:
            next_city = self.find_nearest(self.cities, path[-1])
            path.append(next_city)
            self.cities.remove(next_city)
        return path


class Two_opt(Graph):
    def __init__(self, filename):
        Nodes = Cheapest_insert(filename)
        self.cities = Nodes.find_shortest_path()
        # Nodes = Simulated_annealing(filename)
        # self.cities = Nodes.find_shortest_path()
        # super().__init__(filename)

    def swap(self, route, i, j):
        return route[:i] + route[i:j+1][::-1] + route[j+1:]

    def find_shortest_path(self):
        best = self.cities
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best)-1):
                for j in range(i+1, len(best)):
                    new_route = self.swap(best, i, j)
                    if self.cost(new_route) < self.cost(best):
                        best = new_route
                        improved = True
                self.show_process(i, len(best))
                print('len_path: ', len(best))
        return best


class Cheapest_insert(Graph):
    def __init__(self, filename):
        super().__init__(filename)

    def heuristic(self, cityA, cityB, cityC):
        return self.distance(cityA, cityC) + self.distance(cityB, cityC)\
            - self.distance(cityA, cityB)

    def find_cheapest(self, cities, path):
        cheapest_dis = maxsize

        for i in range(len(path)-1):
            cityA = path[i]
            cityB = path[i+1]

            for cityC in cities:
                heuristic = self.heuristic(cityA, cityB, cityC)
                if heuristic < cheapest_dis:
                    cheapest_dis = heuristic
                    cheapest_city = cityC
                    insert_idx = i
        return cheapest_city, insert_idx

    def insert_inner(self, path, city, i):
        return path[:i+1] + [city] + path[i+1:]

    def insert_outer_right(self, path, city, i):
        if i < len(path) - 2:
            return self.insert_inner(path, city, i)
        else:
            path.append(city)
            return path

    def best_path(self, pathA, pathB):
        if self.cost(pathA) < self.cost(pathB):
            return pathA
        else:
            return pathB

    def find_shortest_path(self):
        path = []
        path.append(self.cities[0])
        path.append(self.cities[1])

        self.cities.pop(0)
        self.cities.pop(1)

        i = 0
        total = len(self.cities)
        while self.cities:
            cheapest_city, insert_idx = self.find_cheapest(self.cities, path)
            pathA = self.insert_inner(path, cheapest_city, insert_idx)
            pathB = self.insert_outer_right(path, cheapest_city, insert_idx)
            path = self.best_path(pathA, pathB)
            self.cities.remove(cheapest_city)
            self.show_process(i, total)
            i += 1
        return path


class Simulated_annealing(Graph):
    def __init__(self, filename):
        # two_opt = Two_opt(filename)
        # self.cities = two_opt.find_shortest_path()
        super().__init__(filename)

    def new_path(self, current_path):
        new_path = current_path[:]
        i = floor(random.random()*len(current_path))
        if i == 0:
            # can not change the start city
            i += 1
        j = floor(random.random()*len(current_path))
        new_path[i], new_path[j] = new_path[j], new_path[i]
        return new_path

    def find_shortest_path(self):
        temp = 10000
        cool_rate = 0.00001

        current_path = self.cities
        best_path = current_path

        while temp > 1:
            new_path = self.new_path(current_path)
            cost_new = self.cost(new_path)
            print('cost new: ', cost_new)  # test
            cost_current = self.cost(current_path)
            print('cost current: ', cost_current)  # test
            sigma = exp((cost_current - cost_new) / temp)
            print('sigma: ', sigma)  # test
            if cost_new < cost_current or sigma > random.random():
                current_path = new_path
                if self.cost(current_path) < self.cost(best_path):
                    best_path = current_path
            temp *= (1 - cool_rate)
            print('cost best: ', self.cost(best_path))  # test
            print('temperature: ', temp)  # test
        return best_path


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
        return sorted(fitness_results.items(),
                      key=operator.itemgetter(1), reverse=True)

    # 'Create a selection function that will be used
    #  to make the list of parent paths'
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
    def find_shortest_path(self, pop_size,
                           elite_size, mutation_rate, generations):
        populations = self.initial_populations(pop_size, self.cities)

        for i in range(generations):
            populations = self.next_generation(
                populations, elite_size, mutation_rate)
            self.show_process(i, generations)

        best_path_index = self.rank_paths(populations)[0][0]
        best_path = populations[best_path_index]
        return best_path

    def find_shortest_pathPlot(self, pop_size,
                               elite_size, mutation_rate, generations):
        populations = self.initial_populations(pop_size, self.cities)
        # pop = self.initial_populations(pop_size, populations)
        progress = []
        progress.append(1 / self.rank_paths(populations)[0][1])

        for i in range(generations):
            populations = self.next_generation(
                populations, elite_size, mutation_rate)
            progress.append(1 / self.rank_paths(populations)[0][1])
            self.show_process(i, generations)

        best_path_index = self.rank_paths(populations)[0][0]
        best_path = populations[best_path_index]

        plt.plot(progress)
        plt.ylabel('Distance')
        plt.xlabel('Generation')
        plt.show()
        return best_path


'''_________________________funtion part______________________________'''


def get_args():
    """config the usage of rsync and arguments"""
    parser = ArgumentParser(prog='TSM')
    parser.add_argument('filename', action='store',
                        help='file to extract cities')
    parser.add_argument('--algo', action='store', default='near',
                        help='Which algorithm to find the shortest path:\
 choose among: near(nearest neighbour), 2opt(two opt), gen(genetic)')
    return parser.parse_args()


def process_algo(algo, filename):
    if algo in ['near', '2opt', 'gen', 'genvisual', 'cheap', 'sim']:
        if algo in 'near':
            path = Nearest(filename)
            nodes = path.find_shortest_path()
        elif algo in '2opt':
            path = Two_opt(filename)
            nodes = path.find_shortest_path()
        elif algo in 'cheap':
            path = Cheapest_insert(filename)
            nodes = path.find_shortest_path()
        elif algo in 'sim':
            path = Simulated_annealing(filename)
            nodes = path.find_shortest_path()
        elif algo in 'gen':
            path = Genetic(filename)
            nodes = path.find_shortest_path(pop_size=100,
                                            elite_size=20,
                                            mutation_rate=0.01,
                                            generations=1500)
        elif algo in 'genvisual':
            path = Genetic(filename)
            nodes = path.find_shortest_pathPlot(pop_size=500,
                                                elite_size=50,
                                                mutation_rate=0.01,
                                                generations=200)
        route = [city.name for city in nodes]
        print(route)
        print('Shortest path length: ', path.cost(nodes))

    else:
        print('Invalid algorithm, select algorithm among:\
 near(nearest neighbour), 2opt(two opt), gen(genetic), cheap')


def main():
    args = get_args()
    algo = args.algo
    filename = args.filename

    # find the shortest by the given algorithm
    process_algo(algo, filename)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("RUNTIME: %s seconds" % (time.time() - start_time))
