import random
import numpy as np
from numpy.random import default_rng
from data_generator import Dataset
from evaluation import evaluation_mre
import matplotlib.pyplot as plt


def identity_algo(data, epsilon, seed=0):
    random.seed(seed)
    rng = default_rng(seed)

    return data + rng.laplace(0.0, 1.0 / epsilon, data.shape)


def uniform_algo(data, epsilon, seed=0):
    random.seed(seed)
    rng = default_rng(seed)

    sum_noise = data.sum() + rng.laplace(0.0, 1.0 / epsilon)
    average_private = sum_noise / data.size

    return np.ones_like(data, dtype=np.float) * average_private


if __name__ == '__main__':
    epsilon = 0.1
    seed = 0
    # Data generation
    data_dimension = 16
    num_center = 3
    sparsity_sigma = 1
    population = 100
    # data = generate_synthetic_dataset_uniform(data_dimension, population)
    dataset = Dataset(data_dimension, population, num_center, sparsity_sigma, seed=0)
    data = dataset.generate_synthetic_dataset_gaussian()
    ax = Dataset.plot_heatmap(data, annotation=True, valfmt ="{x:d}")
    plt.show()
    num_queries = 1000
    query_list = dataset.generate_queries(num_queries)
    print(query_list)

    # identity algorithm
    data_private_idt = identity_algo(data, epsilon, seed=seed)
    Dataset.plot_heatmap(data_private_idt, annotation=False)
    # uniform algorithm
    data_private_uni = uniform_algo(data, epsilon, seed=seed)
    Dataset.plot_heatmap(data_private_uni, annotation=False)

    print(evaluation_mre(data, data_private_idt, query_list))
    # Dataset.plot_data(data_private_idt)
    print(evaluation_mre(data, data_private_uni, query_list))
