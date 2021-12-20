"""
Implementation of K-D tree
- K-D tree standard
- K-D tree cell: Xiao 2010 - Differentially Private Data Release through Multidimensional Partitioning
- K-D tree hybrid: Cormode (2012) - Differentially Private Spatial Decompositions
    - K-D tree uniform, K-D tree geometric, K-D tree hybrid
- HTF: Sina Shaham (2021) - HTF: Homogeneous Tree Framework for Differentially-Private Release of Location Data
"""

import numpy as np
from collections import deque
import random
from numpy.random import default_rng
import matplotlib.pyplot as plt
from dpcomp_core.algorithm import UG


def KDTree_standard(data, epsilon, prt_budget, max_height=16,
                    max_points=32, split_method='median', noise_method='leaf', max_cells = 1000, seed=0):
    if prt_budget:
        epsilon_prt = epsilon * prt_budget
        epsilon_count = epsilon * (1 - prt_budget)
    else:
        epsilon_count = epsilon
        epsilon_prt = None

    kd_tree = KDTree(data, epsilon_prt=epsilon_prt, max_cells = max_cells,
                     max_points=max_points, max_height=max_height, split_method=split_method,
                     height_methods="constant")

    kd_tree.build_tree()

    kd_tree.add_noise(epsilon_count, noise_method=noise_method, seed=seed)

    if noise_method != 'leaf':
        kd_tree.post_weighted_average(kd_tree.root)
        kd_tree.post_mean_consistency()
        kd_tree.post_pruning()
    else:
        kd_tree.post_pruning_leaf(kd_tree.root)

    return kd_tree.generate_private_data(), kd_tree


def KDTree_cell(data, epsilon, c, alpha, H=10, max_height=None, max_points=5, split_method='median',
                height_method='heuristic', seed=0):
    data_ug = UG.UG_engine(c=10).Run(None, data, epsilon=alpha * epsilon, seed=seed)

    kd_tree = KDTree(data_ug, max_points=max_points, max_height=max_height, split_method=split_method,
                     height_methods=height_method, H=H, epsilon_prt=None)
    kd_tree.build_tree()
    kd_tree.add_noise(epsilon * (1 - alpha), noise_method='leaf', seed=seed)
    kd_tree.post_pruning_leaf(kd_tree.root)

    return kd_tree.generate_private_data(), kd_tree


def KDTree_hybrid(data, epsilon, prt_budget, level_switch=3, max_height=16, max_cells = 1000,
                  max_points=5, split_method='median', noise_method='geometric', seed=0):
    if prt_budget:
        epsilon_prt = epsilon * prt_budget
        epsilon_count = epsilon * (1 - prt_budget)
    else:
        epsilon_count = epsilon
        epsilon_prt = None

    if level_switch:
        l = level_switch
    else:
        l = 1E6

    kd_tree = KDTree(data, epsilon_prt=epsilon_prt, l=l, max_cells = max_cells,
                     max_points=max_points, max_height=max_height, split_method=split_method,
                     height_methods="constant")

    kd_tree.build_tree()

    kd_tree.add_noise(epsilon_count, noise_method=noise_method, seed=seed)

    kd_tree.post_weighted_average(kd_tree.root)
    kd_tree.post_mean_consistency()
    kd_tree.post_pruning()

    return kd_tree.generate_private_data(), kd_tree


####################################################################################################################
def find_median(indices_weights, indices):
    sum_all = indices_weights.sum()
    if sum_all % 2 == 0:
        ord1 = np.searchsorted(indices_weights.cumsum(), indices_weights.sum() // 2, side='left')
        ord2 = np.searchsorted(indices_weights.cumsum(), indices_weights.sum() // 2 + 1, side='left')

        return int((indices[ord1] + indices[ord2]) / 2)
    else:
        ord = np.searchsorted(indices_weights.cumsum(), indices_weights.sum() // 2 + 1, side='right')
        return indices[ord]


def find_median_expo(indices_weights, indices, epsilon, flattern=True, seed=0):
    random.seed(seed)
    rng = default_rng(seed)

    # compute median rank
    if indices_weights.sum() == 0:
        indices_weights[indices_weights == 0] = 1
    sum = indices_weights.sum()
    median_rank = sum // 2 + 1 if sum % 2 != 0 else sum // 2

    if flattern:
        # flattern based on weights
        # [1,2,3,4,5]
        # [3,0,5,0,6] => [1,1,1,2,2,2,2,2,5,5,5,5,5,5]
        # rank        => [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        indices_flattern = np.repeat(indices, indices_weights)
        # print(indices_flattern)
        indices_rank = np.arange(1, indices_flattern.size + 1)

        # compute probability based on rank
        indices_prob = np.exp(-1 * epsilon / 2 * np.abs(indices_rank - median_rank))
        indices_prob = indices_prob / np.linalg.norm(indices_prob, ord=1)
        return rng.choice(indices_flattern, p=indices_prob)
    else:
        # rank same for the same value
        # [1,2,3,4,5]
        # [3,0,5,0,6] => [0,3,0,5,0] => cumsum + 1 => Rank [1,4,4,9,9]
        # [1,0,4,0,9]
        indices_weights_shift = np.roll(indices_weights, 1)
        indices_weights_shift[0:1] = 0
        indices_rank = indices_weights_shift.cumsum() + 1

        # compute probability based on rank
        indices_prob = indices_weights * np.exp(-1 * epsilon / 2 * np.abs(indices_rank - median_rank))
        indices_prob = indices_prob / np.linalg.norm(indices_prob, ord=1)

        return rng.choice(indices, p=indices_prob)


def find_mean_noisy(indices_weights, indices, epsilon, seed=0):
    random.seed(seed)
    rng = default_rng(seed)

    # Mean = sum/count
    # impute noise for count - sensitivity of count is 1
    range = indices.max() - indices.min()
    count = indices_weights.sum() + rng.laplace(0.0, 1.0 / epsilon)

    # impute noise for sum - sensitivity of sum is range (hi - low)
    sum = np.dot(indices_weights, indices)
    sum += rng.laplace(0.0, 1.0 / epsilon * range)

    noisy_mean = int(sum / count)

    if noisy_mean < indices.min(): noisy_mean = indices.min()
    if noisy_mean > indices.max(): noisy_mean = indices.max()

    return noisy_mean


##################################################################################################################
# Implementation of K-D Tree
##################################################################################################################
class KDTreeNode:

    def __init__(self, rect: tuple, count: int, height: int, depth: int):
        """
        rect denotes the (x0, y0, h, w) represent area of this node
        """
        self.x0, self.y0, self.h, self.w = rect
        self.height = height
        self.depth = depth
        self.level_odd = 0  # used for flatten the k-d tree
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

    # def add_noisy(self, epsilon: float):
    #     self.epsilon = epsilon
    #     random.seed(seed)
    #     rng = default_rng(seed)
    #     self.count_noisy = self.count + rng.laplace(0.0, 1.0 / epsilon)

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


class KDTree:

    def __init__(self, data, epsilon_prt, split_method='mean', height_methods='constant',
                 max_height: int = None, max_points: int = 1, max_cells = 1000, H=10, l=1E6, seed=0):

        self.root = None
        self.data = data
        self.total_height = 0
        self.num_nodes = 0
        self.seed = seed

        # hybrid => level switch to quad tree
        self.l = l

        # tree height parameters
        self.height_methods = height_methods
        self.max_height = max_height
        self.max_points = max_points
        self.max_cells = max_cells
        self.entropy_threshold = H

        # partition
        self.split_method = split_method
        self.epsilon_prt = None
        if height_methods == 'constant':
            if epsilon_prt and self.max_height:
                self.epsilon_prt = epsilon_prt / self.max_height

        # for post processing
        self.E = None
        self.fanout = 2

    def build_tree(self):
        rect = (0, 0, self.data.shape[0], self.data.shape[1])
        self.root = self.build_tree_helper(rect, height=0)
        self.total_height = self.root.depth

    def build_tree_helper(self, rect: tuple, height: int):
        """
        Build tree from bottom to up
        """
        # base case to exit
        # max height reaches or max_points reaches or cannot divide in coords
        x0, y0, h, w = rect
        # print(rect)
        data = self.data[x0:x0 + h, y0:y0 + w]
        n_count = data.sum()

        unit_condition = (h <= 1 and w <= 1) if height <= self.l else (h <= 1 or w <= 1)

        # do not divide when there is no area or no data points
        if self.height_methods == 'constant':
            if unit_condition or (height + 1 > self.max_height):
                return KDTreeNode(rect, n_count, height=height, depth=0)
        elif self.height_methods == 'heuristic':
            H = np.abs(data - data.mean()).sum()
            if unit_condition or (H < self.entropy_threshold and n_count <= self.max_points) or (n_count <= 1):
                return KDTreeNode(rect, n_count, height=height, depth=0)
        else:
            raise ValueError("Invalid height methods key.")

        # calculate children rects
        if height <= self.l:
            split_axis, split_index = self.calculate_split_point(rect)
            if split_axis == 'x':
                h_local = min(h - 1, max(1, split_index - x0))
                rects = [
                    (x0, y0, h_local, w),
                    (x0 + h_local, y0, h - h_local, w)
                ]
            else:
                w_local = min(w - 1, max(1, split_index - y0))
                rects = [
                    (x0, y0, h, w_local),
                    (x0, y0 + w_local, h, w - w_local)
                ]
        else:  # switch to quad tree
            rects = [
                (x0, y0, h // 2, w // 2),  # upper left
                (x0, y0 + w // 2, h // 2, w - w // 2),  # upper right
                (x0 + h // 2, y0, h - h // 2, w // 2),  # bottom left
                (x0 + h // 2, y0 + w // 2, h - h // 2, w - w // 2),  # bottom right
            ]

        max_depth, children = 0, []
        for local_rect in rects:
            child = self.build_tree_helper(local_rect, height + 1)
            max_depth = max(max_depth, child.depth)
            children.append(child)

        node = KDTreeNode(rect, n_count, height=height, depth=max_depth + 1)
        node.children = children
        self.num_nodes += 1

        return node

    def calculate_split_point(self, rect):
        x0, y0, h, w = rect
        x_indices = np.arange(x0, x0 + h)
        y_indices = np.arange(y0, y0 + w)
        data_array = self.data[x0:x0 + h, y0:y0 + w]
        x_indices_weights = data_array.sum(axis=1)
        y_indices_weights = data_array.sum(axis=0)

        if h <= 1:
            y_split_points = self.find_split_point(y_indices, y_indices_weights)
            return 'y', y_split_points
        if w <= 1:
            x_split_points = self.find_split_point(x_indices, x_indices_weights)
            return 'x', x_split_points

        if data_array.sum() > 0:
            x_indices_mean = np.average(x_indices, weights=x_indices_weights)
            x_indices_var = np.average(np.square(x_indices - x_indices_mean), weights=x_indices_weights)
            y_indices_mean = np.average(y_indices, weights=y_indices_weights)
            x_indices_var = np.average(np.square(x_indices - x_indices_mean), weights=x_indices_weights)
            y_indices_var = np.average(np.square(y_indices - y_indices_mean), weights=y_indices_weights)

            if x_indices_var >= y_indices_var:
                return 'x', self.find_split_point(x_indices, x_indices_weights)
            else:
                return 'y', self.find_split_point(y_indices, y_indices_weights)
        else:
            if h > w:
                return 'x', self.find_split_point(x_indices, x_indices_weights)
            else:
                return 'y', self.find_split_point(y_indices, y_indices_weights)

    def find_split_point(self, indices, indices_weights):

        if self.split_method == 'median':
            if self.epsilon_prt:
                median = find_median_expo(indices_weights, indices, self.epsilon_prt, seed=self.seed)
            else:
                median = find_median(indices_weights, indices)
            return median
        elif self.split_method == 'mean':
            if self.epsilon_prt:
                mean = find_mean_noisy(indices_weights, indices, self.epsilon_prt, seed=self.seed)
            else:
                mean = np.average(indices, weights=indices_weights)
            return int(mean)
        else:
            raise ValueError("Split methods should be one of median and mean.")

    def add_noise(self, epsilon, noise_method='leaf', seed=0):
        """
        Add noise to the tree
        """
        random.seed(seed)
        rng = default_rng(seed)
        # Bread-first search
        q = deque()
        if self.root:
            q.append(self.root)
            while len(q) > 0:
                node = q.popleft()
                # noise adding
                if noise_method == 'leaf':
                    if node.is_leaf():
                        node.epsilon = epsilon
                        node.count_noisy = node.count + rng.laplace(0.0, 1.0 / epsilon)
                elif noise_method == 'uniform':
                    node_path_len = self.total_height + 1
                    node.epsilon = epsilon / node_path_len
                    node.count_noisy = node.count + rng.laplace(0.0, 1.0 / node.epsilon)
                elif noise_method == 'geometric':
                    node.epsilon = 2 ** ((self.total_height - node.depth) / 3) * epsilon * (2 ** (1 / 3) - 1) / (
                            2 ** ((self.total_height + 1) / 3) - 1)
                    node.count_noisy = node.count + rng.laplace(0.0, 1.0 / node.epsilon)

                else:
                    raise ValueError('noise_method should be one of leaf, uniform, geometric')

                # recursion
                for child in node.children:
                    q.append(child)

        # calculate cumsum of E
        E_cumsum = 0
        self.E = []
        for i in range(self.total_height + 1):
            if noise_method == 'geometric':
                epsilon_i = 2 ** ((self.total_height - i) / 6) * epsilon * (2 ** (1 / 6) - 1) / (
                        2 ** ((self.total_height + 1) / 6) - 1)
            elif noise_method == 'uniform':
                epsilon_i = epsilon / (self.total_height + 1)
            else:
                epsilon_i = epsilon

            E_cumsum += (self.fanout ** i) * (epsilon_i ** 2)
            self.E.append(E_cumsum)

    def post_weighted_average(self, node: KDTreeNode):
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

    def post_pruning(self):
        q = deque()
        if self.root:
            q.append(self.root)
            while len(q) > 0:
                node = q.popleft()
                # pruning
                if (node.count_noisy <= self.max_points or node.h*node.w <= self.max_cells) and not node.is_leaf():
                    node.children = []
                else:
                    for child in node.children:
                        q.append(child)

    def post_pruning_leaf(self, node: KDTreeNode):

        if not node.is_leaf():

            # recursion to the end of the tree
            for child in node.children:
                self.post_pruning_leaf(child)

            # weighted average of the node and children
            sum_count_noisy_children = 0
            for child in node.children:
                sum_count_noisy_children += child.count_noisy

            node.count_noisy = sum_count_noisy_children

            if sum_count_noisy_children <= self.max_points or (node.h*node.w <= self.max_cells):
                node.children = []

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
        data_noisy = np.ones_like(self.data, np.float)
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

    epsilon = 0.1
    seed = 1
    # Data generation
    data_dimension = 128
    num_center = 1
    sparsity_sigma = 10
    population = 100000
    # data = generate_synthetic_dataset_uniform(data_dimension, population)
    dataset = Dataset(data_dimension, population, seed=0)
    data = dataset.generate_synthetic_dataset_gaussian(sparsity_sigma=10)
    Dataset.plot_heatmap(data, annotation=False, valfmt="{x:d}")
    # Query workload generation
    num_queries = 1000
    query_list = dataset.generate_queries(num_queries)

    # K-D tree
    data_kdcell, kd_cell = KDTree_cell(data, epsilon, c=10, alpha=0.5, max_points=128, max_height=None,
                                       split_method='median', height_method='heuristic', H=10)

    data_kd_em, kd_em = KDTree_standard(data, epsilon, prt_budget=0.15, max_height=15, split_method='median',
                                        max_points=128, noise_method='geometric')

    data_kd_hybrid, kd_hybrid = KDTree_hybrid(data, epsilon, prt_budget=0.15, level_switch=13, max_height=15,
                                              max_points=128, split_method='median', noise_method='geometric')

    # print(kd_hybrid.total_height)
    # kd_em.draw()
    # kd_hybrid.draw()
    # kd_cell.draw()

    # plt.show()
    print("KD-cell(median): {:.4f}".format(evaluation_mre(data, data_kdcell, query_list)))
    print("KD-em: {:.4f}".format(evaluation_mre(data, data_kd_em, query_list)))
    print("KD-hybrid: {:.4f}".format(evaluation_mre(data, data_kd_hybrid, query_list)))
