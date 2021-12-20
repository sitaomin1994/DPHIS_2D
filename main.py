from random import random
import json
import numpy as np
from numpy.random import default_rng
from data_generator import Dataset
from dpcomp_core.algorithm.AG import AG_engine
from dpcomp_core.algorithm.UG import UG_engine
from evaluation import evaluation_mre
from Algorithm.Baseline import identity_algo, uniform_algo
from Algorithm.KDTree import KDTree_cell
from Algorithm.KDTree_std import KDTree_standard, KDTree_hybrid
from dpcomp_core.algorithm.QuadTree import QuadTree_engine
from Algorithm.HTF import HTF_std
from Algorithm.QuadTree import QuadTree_standard, QuadTree_uniform, QuadTree_geometric
import time

ALGO = {
    'idt':"identity_algo(data, epsilon, seed)",
    'uniform':"uniform_algo(data, epsilon, seed)",
    'AG':"AG_engine(c=10, c2=6, alpha=0.2).Run(None, data, epsilon, seed=seed)",
    'UG':"UG_engine(c=10).Run(None, data, epsilon, seed=seed)",
    'Quad-leaf':"QuadTree_standard(data, epsilon, max_height=8, max_points=128, seed=seed)",
    'Quad-uniform':"QuadTree_uniform(data, epsilon, max_height=8, max_points=128, seed=seed)",
    'Quad-geo':"QuadTree_geometric(data, epsilon, max_height=8, seed=seed)",
    'kd-cell':"KDTree_cell(data, epsilon, c = 10, alpha=0.2, max_points=10, max_height=None, split_method='median', height_method='heuristic', H=10, seed = seed)",
    'kd-uniform':"KDTree_standard(data, epsilon, prt_budget=0.2, max_height=16, split_method='median',noise_method='uniform', max_points=128, seed = seed)",
    'kd-geo':"KDTree_standard(data, epsilon, prt_budget=0.2, max_height=16, split_method='median',noise_method='geometric', max_points=128, seed = seed)",
    'kd-mean': "KDTree_standard(data, epsilon, prt_budget=0.2, max_height=16, split_method='mean',noise_method='geometric', max_points=128, seed = seed)",
    'kd-hybrid': "KDTree_hybrid(data, epsilon, prt_budget=0.2, level_switch=5, max_height=15, max_points=128, split_method='median', noise_method='geometric', seed=seed)",
    'htf': "HTF_std(data, epsilon, epsilon_prt=1E-4, epsilon_height=1E-4, max_points=128, max_cells=1000, c=10, noise_method='geometric', T=7, seed=seed)"
}

TUPLE_RETURN = ['Quad-leaf', 'Quad-uniform', 'Quad-geo', 'kd-cell', 'kd-uniform', 'kd-geo', 'kd-mean','kd-hybrid','htf']

def run_algo(algo_name, datasets, epsilon, seed, query_list):

    start = time.time()
    results = np.zeros(len(datasets))
    for i, data in enumerate(datasets):
        if algo_name in TUPLE_RETURN:
            data_priv,_  = eval(ALGO[algo_name])
        else:
            data_priv = eval(ALGO[algo_name])
        results[i] = evaluation_mre(data, data_priv, query_list)
    end = time.time()

    print('{}: {:4f} time:{:4f}'.format(algo_name, results.mean(), end-start))

    return list(results), (end - start)/100


if __name__ == '__main__':

    # parameters
    epsilon = 0.1
    seed = 1
    rng = default_rng(seed)
    data_dimension = 128
    num_data = 50
    dataset = Dataset(data_dimension, seed=seed)
    num_queries = 1000
    query_list = dataset.generate_queries(num_queries)
    dataset.query_stats(query_list)

    for i, s in enumerate([0.05, 0.1, 0.2, 0.35, 0.5]):

        print("############################################")

        populations = [100000]
        shapes = ['circle']
        sigma = [s]

        dispersion = [s*data_dimension for s in sigma]
        shape_list = rng.choice(shapes, num_data)
        population_list = rng.choice(populations, num_data)

        # dispersion = [2, 4, 8, 10, 20, 40, 60, 80, 100, 120, 160, 200]
        datasets = dataset.generate_random_datasets(dispersion, shape_list, population_list, num_data)
        dataset.data_stats(datasets)
        dataset.plot_data(datasets[10])

        # result
        experiment_result = {
            'params': {
                'epsilon': epsilon,
                'seed': seed,
                'sigma': sigma,
                'shape': shapes,
                'num_data': num_data,
                'population': populations,
                'dimension': data_dimension
            },
            'algos': {
                'idt': {
                    'result': [],
                    'run_time': 0
                },
                'uniform': {
                    'result': [],
                    'run_time': 0
                },
                'AG': {
                    'result': [],
                    'run_time': 0
                },
                'UG': {
                    'result': [],
                    'run_time': 0
                },
                'Quad-leaf': {
                    'result': [],
                    'run_time': 0
                },
                'Quad-uniform': {
                    'result': [],
                    'run_time': 0
                },
                'Quad-geo': {
                    'result': [],
                    'run_time': 0
                },
                'kd-cell': {
                    'result': [],
                    'run_time': 0
                },
                'kd-uniform': {
                    'result': [],
                    'run_time': 0
                },
                'kd-geo': {
                    'result': [],
                    'run_time': 0
                },
                'kd-mean': {
                    'result': [],
                    'run_time': 0
                },
                'kd-hybrid': {
                    'result': [],
                    'run_time': 0
                },
                'htf': {
                    'result': [],
                    'run_time': 0
                }
            },
            'ALGOS': ALGO
        }

        for algo in ALGO.keys():
            result, time_spent = run_algo(algo, datasets, epsilon, seed, query_list)

            experiment_result['algos'][algo]['result'] = result
            experiment_result['algos'][algo]['run_time'] = time_spent

        with open('result_3{:d}.json'.format(i), 'w') as f:
            json.dump(experiment_result, f)
