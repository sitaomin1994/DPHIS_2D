"""
Shaham - HTF: Homogeneous Tree Framework for Differentially-Private Release of Location Data
"""
import math

import numpy as np
from collections import deque
import random
from numpy.random import default_rng
import matplotlib.pyplot as plt
from Algorithm.KDTree import KDTree_cell, KDTree_hybrid, KDTree_standard


def HTF_std(data, epsilon, epsilon_prt, epsilon_height, noise_method='geometric',
            max_points=50, max_cells = 4, c=10, T=3, seed=0):

    htf = HTFTree(data, epsilon, epsilon_prt, epsilon_height, noise_method, max_points, max_cells, c, T, seed)

    htf.build_tree()
    htf.add_noise_recursive(htf.root, 0)
    htf.post_weighted_average(htf.root)
    htf.post_mean_consistency()

    return htf.generate_private_data(), htf


####################################################################################################################
def compute_H(data, curr_index, axis):
    if axis == 0:
        data_left = data[0:curr_index, :]
        data_right = data[curr_index:, :]
    else:
        data_left = data[:, 0:curr_index]
        data_right = data[:, curr_index:]

    H_left = H_right = 0
    if data_left.size != 0:
        H_left = np.abs(data_left - data_left.mean()).sum()

    if data_right.size != 0:
        H_right = np.abs(data_right - data_right.mean()).sum()

    return H_left + H_right


def find_optimal_index_baseline(data, axis, epsilon_prt, seed):
    rng = default_rng(seed)

    range_size = data.shape[axis]
    epsilon_prt_per_index = epsilon_prt / range_size
    opt_index, min_H = None, math.inf

    # Find optimal points which has minimum variance
    for curr_index in range(1, range_size):

        H = compute_H(data, curr_index, axis) + rng.laplace(0.0, 2.0 / epsilon_prt_per_index)
        if H < min_H:
            opt_index, nin_H = curr_index, H

    return opt_index


def find_optimal_index_near_opt(data, axis, T, epsilon_prt, seed):
    rng = default_rng(seed)

    epsilon_prt_per_index = epsilon_prt / (2 * T + 1)
    range_size = data.shape[axis]

    # binary search
    left, right = 1, range_size - 1

    if left == right:  # only range size <= 2
        return left

    k = math.floor((right + left) / 2)

    # compute H_k
    H_k = compute_H(data, k, axis) + rng.laplace(0.0, 2.0 / epsilon_prt_per_index)

    while left <= right and T > 0:
        k1 = math.floor((k + left) / 2)
        k2 = math.floor((right + k) / 2)
        H_k1 = compute_H(data, k1, axis) + rng.laplace(0.0, 2.0 / epsilon_prt_per_index)
        H_k2 = compute_H(data, k2, axis) + rng.laplace(0.0, 2.0 / epsilon_prt_per_index)

        if H_k <= H_k1 and H_k <= H_k2:
            left, right = k1, k2
        elif H_k1 <= H_k and H_k1 <= H_k2:
            right = k
            k = k1
            H_k = H_k1
        else:
            left = k
            k = k2
            H_k = H_k2
        T -= 1

    return k


##################################################################################################################
# Implementation of K-D Tree
##################################################################################################################
class HTFNode:

    def __init__(self, rect: tuple, count: int, height: int, depth: int):
        """
        rect denotes the (x0, y0, h, w) represent area of this node
        """
        self.x0, self.y0, self.h, self.w = rect
        self.height = height
        self.depth = depth
        # Related to noise
        self.count = count
        self.epsilon = None
        self.count_noisy = count
        # children of KD-tree
        self.children = []

    def __repr__(self):
        return str((self.x0, self.y0, self.x0 + self.w, self.y0 + self.h)) + "count:{}".format(self.count)

    def __str__(self):
        return '({:.2f}, {:.2f}, {:.2f}, {:.2f}) - count: {:d}, height: {:d}, depth: {:d}'.format(
            self.x0, self.y0, self.x0 + self.h, self.y0 + self.w, self.count, self.height, self.depth
        )

    def is_leaf(self):
        return len(self.children) == 0

    def intersects(self, other):
        """Does Rect object other intersect this Rect?"""
        return not (other.x0 > self.x0 + self.h or
                    other.x0 + other.h < self.x0 or
                    other.y0 > self.y0 + self.w or
                    other.y0 + other.w < self.y0)

    def draw(self, ax, c='r', lw=1, **kwargs):
        x1, y1 = self.x0, self.y0
        x2, y2 = self.x0 + self.h, self.y0 + self.w
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], c=c, lw=lw, **kwargs)


class HTFTree:

    def __init__(self, data, epsilon, epsilon_prt, epsilon_height,
                 noise_method='geometric', max_points=32, max_cells = 4, c=10, T=3, seed=0):

        self.root = None
        self.data = data
        self.total_height = 0
        self.num_nodes = 0
        self.seed = seed

        random.seed(self.seed)
        rng = default_rng(self.seed)

        # tree height parameters - calculate height from number of data points
        self.epsilon_height = epsilon_height
        self.max_height = int(np.log2((self.data.sum() + rng.laplace(0.0, 1.0 / self.epsilon_height))*epsilon / c))
        if self.max_height < int(np.log2(data.shape[0]*data.shape[1])):
            self.max_height = int(np.log2(data.shape[0]*data.shape[1]))
        # self.max_height = 15

        # partition - epsilon of partition for each level
        self.epsilon_prt = epsilon_prt
        self.T = T

        # data noisy privacy budge
        self.epsilon_count = epsilon - epsilon_height - epsilon_prt * self.max_height
        self.noise_method = noise_method
        self.max_points = max_points
        self.max_unit = int(math.sqrt(max_cells))

        #print(self.epsilon_count, self.max_height)

        # for post processing
        self.fanout = 2
        self.E = self.calculate_E()

    def build_tree(self):

        rect = (0, 0, self.data.shape[0], self.data.shape[1])
        self.root = self.build_tree_helper(rect, height=0)
        self.total_height = self.root.depth

    def build_tree_helper(self, rect: tuple, height: int):
        """
        Build tree from bottom to up
        """
        x0, y0, h, w = rect
        data = self.data[x0:x0 + h, y0:y0 + w]
        n_count = data.sum()

        # do not divide when meet max height
        # base case to exit
        if (h <= 1 and w <= 1) or (height + 1 > self.max_height):
            return HTFNode(rect, n_count, height=height, depth=0)

        # calculate optimal split points
        if h == 1:
            axis = 1
        elif w == 1:
            axis = 0
        else:
            axis = 0 if height % 2 else 1

        split_index = self.get_optimal_split_point(rect, axis)

        if axis == 0:
            h_local = split_index - x0
            rects = [
                (x0, y0, h_local, w),
                (x0 + h_local, y0, h - h_local, w)
            ]
        else:
            w_local = split_index - y0
            rects = [
                (x0, y0, h, w_local),
                (x0, y0 + w_local, h, w - w_local)
            ]

        max_depth, children = 0, []
        for local_rect in rects:
            child = self.build_tree_helper(local_rect, height + 1)
            max_depth = max(max_depth, child.depth)
            children.append(child)

        node = HTFNode(rect, n_count, height=height, depth=max_depth + 1)
        node.children = children
        self.num_nodes += 1

        return node

    def get_optimal_split_point_baseline(self, rect, axis):

        x0, y0, h, w = rect
        data = self.data[x0:x0 + h, y0:y0 + w]

        opt_index = find_optimal_index_baseline(data, axis, epsilon_prt=self.epsilon_prt, seed=self.seed)
        return x0 + opt_index if axis == 0 else y0 + opt_index

    def get_optimal_split_point(self, rect, axis):

        x0, y0, h, w = rect
        data = self.data[x0:x0 + h, y0:y0 + w]
        opt_index = find_optimal_index_near_opt(data, axis, T=self.T, epsilon_prt=self.epsilon_prt, seed=self.seed)
        return x0 + opt_index if axis == 0 else y0 + opt_index

    def add_noise_recursive(self, node: HTFNode, epsilon_accu):

        rng = default_rng(self.seed)
        if self.noise_method == 'uniform':
            epsilon_node = self.epsilon_count / (node.height + node.depth + 1)
        elif self.noise_method == 'geometric':
            epsilon_node = 2 ** ((self.total_height - node.depth) / 3) * self.epsilon_count * (2 ** (1 / 3) - 1) / (
                    2 ** ((self.total_height + 1) / 3) - 1)
        else:
            raise ValueError('[Noise Method] is not supported.')

        # budget overload, prune the branch
        if epsilon_accu + epsilon_node >= self.epsilon_count:
            epsilon_node = self.epsilon_count - epsilon_accu
            node.epsilon = epsilon_node
            node.count_noisy = node.count + rng.laplace(0.0, 1 / epsilon_node)
            node.children = []
            return

        epsilon_accu += epsilon_node
        node.epsilon = epsilon_node
        node.count_noisy = node.count + rng.laplace(0.0, 1 / epsilon_node)

        # pruning
        if (node.count_noisy <= self.max_points) or (node.h*node.w <= self.max_unit) or \
                epsilon_accu == self.epsilon_count:
            epsilon_remain = self.epsilon_count - epsilon_accu
            if epsilon_remain > 0:
                node.count_noisy = node.count + rng.laplace(0.0, 1 / epsilon_remain)
            node.children = []
        else:
            for child in node.children:
                self.add_noise_recursive(child, epsilon_accu)

    def calculate_E(self):

        E_cumsum = 0
        E = []
        for i in range(self.max_height + 1):
            if self.noise_method == 'geometric':
                epsilon_i = 2 ** ((self.total_height - i) / 3) * self.epsilon_count * (2 ** (1 / 3) - 1) / (
                        2 ** ((self.total_height + 1) / 3) - 1)
            elif self.noise_method == 'uniform':
                epsilon_i = self.epsilon_count / (self.total_height + 1)
            else:
                raise ValueError("Invalid noise method.")

            E_cumsum += (self.fanout ** i) * (epsilon_i ** 2)
            E.append(E_cumsum)

        return E

    def post_weighted_average(self, node: HTFNode):
        '''
        First postprocessing: Weighted averaging the count between node and his children
        Hay 2020 - Boosting the Accuracy of Differentially Private Histograms Through Consistency
        '''
        if not node.is_leaf():

            # recursion to the end of the tree
            for child in node.children:
                self.post_weighted_average(child)

            # weighted average of the node and children
            eps1 = node.epsilon  # self privacy budget
            sum_count_noisy_children = 0
            eps2 = 0  # max privacy budget for the children
            for child in node.children:
                sum_count_noisy_children += child.count_noisy
                eps2 = max(eps2, child.epsilon)

            alpha = (self.E[node.depth] - self.E[node.depth - 1]) / self.E[node.depth]
            # alpha = (4*eps1**2)/(4*eps1**2 + eps2**2)
            node.count_noisy = alpha * node.count_noisy + (1 - alpha) * sum_count_noisy_children

    def post_mean_consistency(self):
        """
        Add noise to the tree
        """
        q = deque()
        if self.root:
            q.append(self.root)
            while len(q) > 0:
                node = q.popleft()
                # noise adding
                if not node.is_leaf():
                    sum_count_child = sum(child.count_noisy for child in node.children)
                    for child in node.children:
                        child.count_noisy += (node.count_noisy - sum_count_child) / len(node.children)

                for child in node.children:
                    q.append(child)

    def check_consistency(self):
        """
        Add noise to the tree
        """
        consistency = True
        total_diff = 0
        q = deque()
        if self.root:
            q.append(self.root)
            while len(q) > 0:
                node = q.popleft()
                # noise adding
                if not node.is_leaf():
                    sum_count_child = sum(child.count_noisy for child in node.children)
                    if sum_count_child != node.count_noisy:
                        consistency = False
                        total_diff += abs(node.count_noisy - sum_count_child)
                # recursion
                for child in node.children:
                    q.append(child)

        return consistency, total_diff

    def generate_private_data(self):
        """
        Using BFS
        """
        data_noisy = np.zeros_like(self.data,dtype = np.float)
        q = deque()
        if self.root:
            q.append(self.root)
            while len(q) > 0:
                node = q.popleft()
                if node.is_leaf():
                    data_noisy[node.x0:node.x0 + node.h, node.y0:node.y0 + node.w] = node.count_noisy / (
                            node.h * node.w)
                else:
                    for child in node.children:
                        q.append(child)
        return data_noisy

    def draw(self, data=None):
        """We use bread first search to traverse the tree draw all boxes"""
        if data is None:
            data = self.data
        data_dimension = data.shape[0]
        DPI = 144
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

        q = deque()
        if self.root:
            q.append(self.root)
            while len(q) > 0:
                node = q.popleft()
                node.draw(ax)
                if len(node.children) > 0:
                    for child in node.children:
                        q.append(child)
        plt.show()


if __name__ == '__main__':
    # unit test for k-d tree
    from data_generator import Dataset
    from evaluation import evaluation_mre, evaluation_mae

    epsilon = 1
    seed = 1
    # Data generation
    data_dimension = 128
    num_center = 1
    sparsity_sigma = 30
    population = 100000
    # data = generate_synthetic_dataset_uniform(data_dimension, population)
    dataset = Dataset(data_dimension, population, seed=0)
    data = dataset.generate_synthetic_dataset_gaussian(sparsity_sigma=sparsity_sigma)
    Dataset.plot_heatmap(data, annotation=False, valfmt="{x:d}")
    # Query workload generation
    num_queries = 1000
    query_list = dataset.generate_queries(num_queries)

    # K-D tree
    # data_kdcell, kd_cell = KDTree_cell(data, epsilon, gz=2, alpha=0.5, max_points=50, max_height=None,
    #                                    split_method='median', height_method='heuristic', H=10, seed=seed)
    #
    # data_kd_em, kd_em = KDTree_standard(data, epsilon, prt_budget=0.1, max_height=15, split_method='median',
    #                                     noise_method='uniform', post_process=True, seed=seed)
    #
    # data_kd_hybrid, kd_hybrid = KDTree_hybrid(data, epsilon, prt_budget=0.1, level_switch=4, max_height=15,
    #                                           split_method='median', noise_method='uniform', post_process=True,
    #                                           seed=seed)

    data_htf, htf_std = HTF_std(data, epsilon, epsilon_prt=1E-4, epsilon_height=1E-4, max_points= 100, max_cells= 1000,
                                noise_method='geometric', T=7, seed=seed)

    print(data_htf.sum())
    print(htf_std.total_height)
    print(htf_std.max_height)
    # htf_std.draw()
    # kd_hybrid.draw()
    # kd_cell.draw()

    # plt.show()
    # print("KD-cell(median): {:.4f}".format(evaluation_mre(data, data_kdcell, query_list)))
    # print("KD-em: {:.4f}".format(evaluation_mre(data, data_kd_em, query_list)))
    # print("KD-hybrid: {:.4f}".format(evaluation_mre(data, data_kd_hybrid, query_list)))
    print("HTF: {:.4f}".format(evaluation_mre(data, data_htf, query_list)))
