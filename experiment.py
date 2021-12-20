import numpy as np
import random
from numpy.random import default_rng
import matplotlib.pyplot as plt

from data_generator import Dataset
from evaluation import compute_all_queries, evaluation_mre, evaluation_mae

from Algorithm.Baseline import identity_algo, uniform_algo
from Algorithm.UniformGrids import UniformGrids, AdaptiveGrids, plot_grids
from Algorithm.QuadTree import QuadTree_standard, QuadTree_uniform, QuadTree_geometric
from Algorithm.KDTree import KDTree_standard, KDTree_hybrid, KDTree_cell
from Algorithm.HTF import HTF_std
from dpcomp_core.algorithm import AG
from dpcomp_core.algorithm import UG
from dpcomp_core.algorithm import QuadTree
from dpcomp_core.algorithm import privelet2D
from dpcomp_core.algorithm import HB2D

if __name__ == '__main__':
    epsilon = 0.1
    seed = 1
    # Data generation
    data_dimension = 256
    num_center = 1
    sparsity_sigma = 10
    population = 500000
    # data = generate_synthetic_dataset_uniform(data_dimension, population)
    dataset = Dataset(data_dimension, population, seed=0)
    data = dataset.generate_dataset_1D_gaussian(num_center = num_center, sparsity_sigma=sparsity_sigma)
    Dataset.plot_heatmap(data, annotation=False, valfmt="{x:d}")

    # Query workload generation
    num_queries = 1000
    query_list = dataset.generate_queries(num_queries)
    print(query_list)
    # Compute answer of the query
    print(compute_all_queries(data, query_list))

    # identity algorithm
    data_private_idt = identity_algo(data, epsilon, seed=seed)
    #dataset.plot_heatmap(data_private_idt)
    # uniform algorithm
    data_private_uni = uniform_algo(data, epsilon, seed=seed)
    # UG
    data_private_ug1 = UG.UG_engine(c=10).Run(None,data, epsilon, seed = seed)
    data_private_ug, ug_grids = UniformGrids(data, c=10, epsilon=epsilon, num_grids=None, seed=seed)
    # AG
    data_private_ag1 = AG.AG_engine(c=10, c2 = 5, alpha=0.4).Run(None, data,epsilon=epsilon, seed=seed)
    data_private_ag, ag_grids = AdaptiveGrids(data, c1=10, c2=5, alpha=0.4, epsilon=epsilon, post_process=True,
                                              seed=seed)
    # Quad tree standard
    data_private_qtree = QuadTree.QuadTree_engine().Run(None, data, epsilon, seed = seed)
    data_private_quad_std, quad_tree = QuadTree_standard(data, epsilon, max_height=12, max_points=1, seed=seed)
    data_private_quad_uniform, quad_tree_uniform = QuadTree_uniform(data, epsilon, max_height=12, max_points=1,
                                                                    seed=seed)
    data_private_quad_geometric, quad_tree_geometric = QuadTree_geometric(data, epsilon, max_height=12, max_points=1,
                                                                          seed=seed)

    # KD TREE
    data_kdcell, kd_cell = KDTree_cell(data, epsilon, gz=2, alpha=0.5, max_points=50, max_height=None,
                                       split_method='median', height_method='heuristic', H=10)

    data_kd_em, kd_em = KDTree_standard(data, epsilon, prt_budget=0.1, max_height=15, split_method='median',
                                        noise_method='uniform', post_process=True)

    data_kd_hybrid, kd_hybrid = KDTree_hybrid(data, epsilon, prt_budget=0.1, level_switch=3, max_height=15,
                                              split_method='median', noise_method='uniform', post_process=True)

    data_htf, htf_std = HTF_std(data, epsilon, epsilon_prt=1E-4, epsilon_height=1E-4, max_points=100, c = 12,
                                noise_method='geometric', T=1, seed=seed)

    data_private_privlet = privelet2D.privelet2D_engine().Run(None, data, epsilon, seed)
    data_private_hb = HB2D.HB2D_engine().Run(None, data, epsilon, seed)
    kd_em.draw()
    plot_grids(data, ag_grids)
    # plt.show()

    print("Identity: {:.4f}".format(evaluation_mre(data, data_private_idt, query_list)))
    print("Uniform: {:.4f}".format(evaluation_mre(data, data_private_uni, query_list)))
    print("UG: {:.4f}".format(evaluation_mre(data, data_private_ug, query_list)))
    print("UG1: {:.4f}".format(evaluation_mre(data, data_private_ug1, query_list)))
    print("AG: {:.4f}".format(evaluation_mre(data, data_private_ag, query_list)))
    print("AG1: {:.4f}".format(evaluation_mre(data, data_private_ag1, query_list)))
    print("Quad: {:.4f}".format(evaluation_mre(data, data_private_qtree, query_list)))
    print("Quad-std: {:.4f}".format(evaluation_mre(data, data_private_quad_std, query_list)))
    print("Quad-uniform: {:.4f}".format(evaluation_mre(data, data_private_quad_uniform, query_list)))
    print("Quad-geometric: {:.4f}".format(evaluation_mre(data, data_private_quad_geometric, query_list)))
    print("KD-cell(median): {:.4f}".format(evaluation_mre(data, data_kdcell, query_list)))
    print("KD-em: {:.4f}".format(evaluation_mre(data, data_kd_em, query_list)))
    print("KD-hybrid: {:.4f}".format(evaluation_mre(data, data_kd_hybrid, query_list)))
    print("HTF: {:.4f}".format(evaluation_mre(data, data_htf, query_list)))
    print("Privlet: {:.4f}".format(evaluation_mre(data, data_private_privlet, query_list)))
    print("HB: {:.4f}".format(evaluation_mre(data, data_private_hb, query_list)))

    # Dataset.plot_data(data_private_idt)
    # Dataset.plot_heatmap(data_private_uni)
    # Dataset.plot_data(data_private_ug)
    # plot_grids(data, ag_grids)
    # Dataset.plot_heatmap(data_private_ag)
    # Dataset.plot_data(data_private_quad_std)
