"""
UG, AG (Qardaji 2013) - Differentially private grids for geospatial data
"""
import numpy as np
import math
import random as random
from numpy.random import default_rng
import matplotlib.pyplot as plt


def UniformGrids(data: np.ndarray, c: float = 10, epsilon: float = 1.0, num_grids=None, seed=0):
    """
    Uniform grid algorithm
    """

    random.seed(seed)
    rng = default_rng(seed)

    data_noisy = np.ones_like(data, dtype = np.float)
    N = data.sum()
    dimension = data.shape[0]  # we assume a square region
    if num_grids is None:
        num_grids = int(math.sqrt(N * epsilon / c))  # grid num from paper
    if num_grids < 1: num_grids = 1
    if num_grids > dimension: num_grids = dimension

    grid_size = round(dimension // num_grids)

    # generate grid
    grids = []
    x0, y0, x_size, y_size = 0, 0, 0, 0
    for i in range(0, dimension, grid_size):
        for j in range(0, dimension, grid_size):
            x0 = i
            y0 = j
            x_size = dimension - x0 if x0 + grid_size > dimension else grid_size
            y_size = dimension - y0 if y0 + grid_size > dimension else grid_size

            grids.append((x0, y0, x_size, y_size))

    # for last grids
    if x0 + x_size < dimension and y0 + y_size < dimension:
        grids.append((x0 + x_size, y0 + y_size, dimension, dimension))

    # Adding laplace noise to the grids
    for cell in grids:
        x0, y0, x_size, y_size = cell
        count = data[x0: x0 + x_size, y0:y0 + y_size].sum()
        average = (count + rng.laplace(0.0, 1.0 / epsilon)) / (x_size * y_size)
        data_noisy[x0: x0 + x_size, y0: y0 + y_size] = average

    return data_noisy, grids


def AdaptiveGrids(data: np.ndarray, c1: int = 10, c2: int = 5, alpha: float = 0.5, epsilon: float = 0.1,
                  post_process=True, num_grids_level1=None, seed: int = 0):
    """
    Algorithm for adaptive grids
    """
    # random seed
    random.seed(seed)
    rng = default_rng(seed)

    # data returned
    data_noisy = np.ones_like(data, dtype = np.float)

    # Uniform grid first level
    N = data.sum()  # number of counts in data
    dimension = data.shape[0]  # we assume a square region

    # first level grids
    if num_grids_level1 is None:
        num_grids_level1 = max(10, int(1 / 4 * math.sqrt(N * epsilon / c1)))  # grid size of level 1
    if num_grids_level1 < 1: num_grids_level1 = 1
    if num_grids_level1 > dimension: num_grids_level1 = dimension
    print(num_grids_level1)
    grid_size_level1 = round(dimension / num_grids_level1)

    grids = []
    for i in range(0, dimension, grid_size_level1):
        for j in range(0, dimension, grid_size_level1):
            # first level grids partition
            x0 = i
            y0 = j
            x_size = dimension - x0 if x0 + grid_size_level1 > dimension else grid_size_level1
            y_size = dimension - y0 if y0 + grid_size_level1 > dimension else grid_size_level1

            grids.append((x0, y0, x_size, y_size))
            # print((x0,y0, x_size, y_size))

            # add noisy to first level grids
            n_count = data[x0:x0 + x_size, y0:y0 + y_size].sum()
            n_count_noisy = (n_count + rng.laplace(0.0, 1.0 / (alpha * epsilon)))
            data_noisy[x0:x0 + x_size, y0:y0 + y_size] = n_count_noisy / (x_size * y_size)

            # second level grids
            if n_count_noisy > 1:
                num_grids_level2 = math.ceil(math.sqrt(n_count_noisy * (1 - alpha) * epsilon / c2))
                if num_grids_level2 < 1: num_grids_level2 = 1
                if num_grids_level2 > x_size: num_grids_level2 = x_size
                grid_size_level2 = round(x_size / num_grids_level2)
            else:
                grid_size_level2 = x_size

            sum_n_count_noisy_local = 0
            local_grids = []
            local_noises = []
            for p in range(0, x_size, grid_size_level2):
                for q in range(0, y_size, grid_size_level2):
                    # second level grids partition
                    x0_local = x0 + p
                    y0_local = y0 + q
                    x_size_local = x0 + x_size - x0_local if x0_local + grid_size_level2 > x0 + x_size else grid_size_level2
                    y_size_local = y0 + y_size - y0_local if y0_local + grid_size_level2 > y0 + y_size else grid_size_level2

                    local_grids.append((x0_local, y0_local, x_size_local, y_size_local))

                    # add noise to second level grids
                    n_count_local = data_noisy[x0_local:x0_local + x_size_local, y0_local:y0_local + y_size_local].sum()
                    n_count_noisy_local = n_count_local + rng.laplace(0.0, 1.0 / ((1 - alpha) * epsilon))
                    local_noises.append(n_count_noisy_local)
                    # calculate sum(ui,j) - sum of noise count in second level grids
                    sum_n_count_noisy_local += n_count_noisy_local

            # post-processing => make first level count to be consistent with second level count:
            # sum_n_count_noisy <-> n_count_noisy
            # weighted average
            m2 = grid_size_level2
            a = (alpha ** 2) * (m2 ** 2) / ((1 - alpha) ** 2 + (alpha ** 2) * (m2 ** 2))
            n_count_noisy = a * n_count_noisy + (1 - a) * sum_n_count_noisy_local

            diff = (n_count_noisy - sum_n_count_noisy_local) / len(local_grids)

            for local_grid, local_noisy in zip(local_grids, local_noises):
                x0_local, y0_local, x_size_local, y_size_local = local_grid
                data_noisy[x0_local:x0_local + x_size_local, y0_local:y0_local + y_size_local] = \
                    (local_noisy + diff) / (x_size_local * y_size_local)
                if x_size_local == 0 or y_size_local == 0:
                    print(local_grid)

            # add new counts based on post-processing
            # data_noisy[x0:x0 + x_size, y0:y0 + y_size] = n_count_noisy / (x_size * y_size)
            grids.extend(local_grids)

    return data_noisy, grids


def plot_grids(data, grids: list, c='r', lw=1, **kwargs):
    # scatter plot
    DPI = 144
    data_dimension = data.shape[0]
    fig = plt.figure(figsize=(700 / DPI, 700 / DPI), dpi=DPI)
    ax = plt.subplot()
    ax.set_xlim(0, data_dimension)
    ax.set_ylim(0, data_dimension)

    points_x, points_y, counts = [], [], []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] > 0:
                points_x.append(i)
                points_y.append(j)
                counts.append(data[i, j])

    ax.scatter(points_x, points_y, s=30, c=counts)

    for grid in grids:
        x0, y0, h, w = grid
        x1, y1 = x0, y0
        x2, y2 = x0 + h, y0 + w
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], c=c, lw=lw, **kwargs)

    plt.show()


if __name__ == '__main__':
    from data_generator import Dataset
    from QuadTree import QuadTree_geometric
    from evaluation import evaluation_mre
    from dpcomp_core.algorithm.UG import UG_engine
    from dpcomp_core.algorithm.AG import AG_engine

    # unit test for quad tree
    epsilon = 0.1
    seed = 1
    # Data generation
    data_dimension = 64
    num_center = 30
    sparsity_sigma = 3
    population = 1000
    # data = generate_synthetic_dataset_uniform(data_dimension, population)
    dataset = Dataset(data_dimension, population, seed=0)
    data = dataset.generate_synthetic_dataset_gaussian(sparsity_sigma=10)
    Dataset.plot_heatmap(data, annotation=False, valfmt="{x:d}")
    # Query workload generation
    num_queries = 1000
    query_list = dataset.generate_queries(num_queries)
    # Compute answer of the query

    # Quad tree standard
    # data_private_quad_std, quad_tree = QuadTree_standard(data, epsilon, max_height=None, max_points=5, seed=seed)
    data_private_ug, ug = UniformGrids(data, c=10, epsilon = epsilon, seed = seed)
    data_private_dp_ug = UG_engine(c = 10).Run(None, data, epsilon, seed)
    #plot_grids(data, ag)
    data_private_ag, ag = AdaptiveGrids(data, c1=10, c2 = 5, alpha= 0.4, epsilon=epsilon, seed=seed)
    plot_grids(data, ag)
    data_private_dp_ag = AG_engine(c = 10,c2=5, alpha=0.4).Run(None,data, epsilon, seed)

    data_private_quad_geometric, quad_tree_geometric = QuadTree_geometric(data, epsilon, max_height=8, max_cells=128,
                                                                          max_points=128, seed=seed)

    # quad_tree_geometric.draw()
    # quad_tree_uniform.draw()
    # print(quad_tree_uniform.total_height)
    # plot_grids(ag_grids, ax)
    # plt.show()
    # print("Quad-std: {:.4f}".format(evaluation_mre(data, data_private_quad_std, query_list)))
    print("UG: {:.4f}".format(evaluation_mre(data, data_private_ug, query_list)))
    print("UG: {:.4f}".format(evaluation_mre(data, data_private_dp_ug, query_list)))
    print("AG: {:.4f}".format(evaluation_mre(data, data_private_ag, query_list)))
    print("AG: {:.4f}".format(evaluation_mre(data, data_private_dp_ag, query_list)))
    print("Quad-geometric: {:.4f}".format(evaluation_mre(data, data_private_quad_geometric, query_list)))
