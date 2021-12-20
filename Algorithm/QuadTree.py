"""
Cormode (2012) - Differentially Private Spatial Decompositions
- QuadTree-standard (no noise added) - baseline
- QuadTree-uniform (uniform budget)
- QuadTree-geo (Geometric budget)
- QuadTree-post (Post-processing)
- QuadTree-opt (Combine both geo and post-processing)
"""
import numpy as np
from collections import deque
from numpy.random import default_rng
import random as random
from Algorithm.UniformGrids import UniformGrids
from data_generator import Dataset
from evaluation import evaluation_mre
import matplotlib.pyplot as plt
from dpcomp_core.algorithm.QuadTree import QuadTree_engine
from dpcomp_core.algorithm import UG

seed = 0


def QuadTree_standard(data, epsilon, max_points=1, max_height=None, max_cells = 128, seed: int = 0):

    quad_tree = QuadTree(data, epsilon, max_points=max_points, max_cells=max_cells,
                         max_height=max_height, noise_method='leaf')
    quad_tree.build_tree()

    quad_tree.add_noise(epsilon, seed=seed)
    quad_tree.post_pruning_leaf(quad_tree.root)
    return quad_tree.generate_private_data(), quad_tree


def QuadTree_uniform(data, epsilon, max_points=32, max_height=8, max_cells = 128, seed: int = 0):

    quad_tree = QuadTree(data, epsilon , height_methods='constant',noise_method='uniform',
                         max_height=max_height, max_points=max_points, max_cells=max_cells)
    quad_tree.build_tree()
    quad_tree.add_noise(epsilon=epsilon, seed=seed)

    quad_tree.post_weighted_average(quad_tree.root)
    quad_tree.post_mean_consistency()
    quad_tree.post_pruning()

    return quad_tree.generate_private_data(), quad_tree


def QuadTree_geometric(data, epsilon, max_points=32, max_height=8, max_cells = 128, seed: int = 0):

    quad_tree = QuadTree(data, epsilon, height_methods='constant',noise_method='geometric',
                         max_height=max_height, max_points=max_points, max_cells=max_cells)
    quad_tree.build_tree()
    quad_tree.add_noise(epsilon=epsilon, seed=seed)

    quad_tree.post_weighted_average(quad_tree.root)
    quad_tree.post_mean_consistency()
    quad_tree.post_pruning()

    return quad_tree.generate_private_data(), quad_tree


#################################################################################
# Quad Tree Implementation
#################################################################################
class TreeNode:

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
        # children of quadtree follow upper left, upper right, bottom left bottom right
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


class QuadTree:

    def __init__(self, data: np.ndarray, epsilon, height_methods='constant',noise_method = 'leaf',
                 max_height: int = None, max_points: int = 1, max_cells = 1000,H_entropy_thres=10, seed = 1):

        self.root = None
        self.max_height = max_height
        self.max_points = max_points
        self.max_cells = max_cells
        self.total_height = 0
        self.num_nodes = 0
        self.data = data
        self.height_methods = height_methods
        self.entropy_threshold = H_entropy_thres
        self.noise_method = noise_method
        self.rng = default_rng(seed)
        # for post processing
        self.epsilon = epsilon
        self.E = None
        self.fanout = 4

    def build_tree(self):
        rect = (0, 0, self.data.shape[0], self.data.shape[1])
        self.root = self.build_tree_recursive(rect, height=0)
        self.total_height = self.root.depth

        # calculate cumsum of E
        E_cumsum = 0
        self.E = []
        for i in range(self.total_height + 1):
            if self.noise_method == 'geometric':
                epsilon_i = 2 ** ((self.total_height - i) / 3) * self.epsilon * (2 ** (1 / 3) - 1) / (
                        2 ** ((self.total_height + 1) / 3) - 1)
            elif self.noise_method == 'uniform':
                epsilon_i = self.epsilon / (self.total_height + 1)
            else:
                epsilon_i = self.epsilon

            E_cumsum += (4 ** i) * (epsilon_i ** 2)
            self.E.append(E_cumsum)

    def build_tree_recursive(self, rect: tuple, height):
        """
        Build tree from bottom to up
        """
        # base case to exit
        # max height reaches or max_points reaches or cannot divide in coords
        x0, y0, h, w = rect
        data = self.data[x0:x0 + h, y0:y0 + w]
        n_count = data.sum()

        # do not divide when there is no area or no data points
        if (h <= 1 or w <= 1) or (height + 1 > self.max_height):
            #print(x0,y0,h,w)
            return TreeNode(rect, n_count, height=height, depth=0)

        rects = [
            (x0, y0, h // 2, w // 2),  # upper left
            (x0, y0 + w // 2, h // 2, w - w // 2),  # upper right
            (x0 + h // 2, y0, h - h // 2, w // 2),  # bottom left
            (x0 + h // 2, y0 + w // 2, h - h // 2, w - w // 2),  # bottom right
        ]
        max_depth, children = 0, []
        for local_rect in rects:
            child = self.build_tree_recursive(local_rect, height + 1)
            max_depth = max(max_depth, child.depth)
            children.append(child)

        node = TreeNode(rect, n_count, height=height, depth=max_depth + 1)
        node.children = children
        self.num_nodes += 1

        return node

    def add_noise(self, epsilon, seed=0):
        """
        Add noise to the tree
        """
        noise_method = self.noise_method

        q = deque()
        if self.root:
            q.append(self.root)
            while len(q) > 0:
                node = q.popleft()
                # noise adding
                if noise_method == 'leaf':
                    if node.is_leaf():
                        node.epsilon = epsilon
                        node.count_noisy = node.count + self.rng.laplace(0.0, 1.0 / epsilon)
                elif noise_method == 'uniform':
                    node_path_len = self.total_height + 1
                    node.epsilon = epsilon / node_path_len
                    node.count_noisy = node.count + self.rng.laplace(0.0, 1.0 / node.epsilon)
                elif noise_method == 'geometric':
                    node.epsilon = 2 ** ((self.total_height - node.depth) / 3) * epsilon * (2 ** (1 / 3) - 1) / (
                            2 ** ((self.total_height + 1) / 3) - 1)
                    node.count_noisy = node.count + self.rng.laplace(0.0, 1.0 / node.epsilon)

                else:
                    raise ValueError('noise_method should be one of leaf, uniform, geometric')

                # recursion
                for child in node.children:
                    q.append(child)

    def post_weighted_average(self, node: TreeNode):
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

            #alpha = (self.E[node.depth] - self.E[node.depth - 1])/self.E[node.depth]
            alpha = (4*eps1**2)/(4*eps1**2 + eps2**2)
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
                        child.count_noisy += (node.count_noisy - sum_count_child)*1.0 / len(node.children)

                for child in node.children:
                    q.append(child)

    def post_pruning(self):
        q = deque()
        if self.root:
            q.append(self.root)
            while len(q) > 0:
                node = q.popleft()
                # pruning
                # print(node.count_noisy, threshold)
                if (node.count_noisy <= self.max_points or node.h*node.w <= self.max_cells) and not node.is_leaf():
                    node.children = []
                else:
                    for child in node.children:
                        q.append(child)

    def post_pruning_leaf(self, node:TreeNode):

        if not node.is_leaf():

            # recursion to the end of the tree
            for child in node.children:
                self.post_pruning_leaf(child)

            # weighted average of the node and children
            sum_count_noisy_children = 0
            for child in node.children:
                sum_count_noisy_children += child.count_noisy

            node.count_noisy = sum_count_noisy_children

            if sum_count_noisy_children <= self.max_points or node.h*node.w <= self.max_cells:
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
        Using depth first search
        """
        data_noisy = np.zeros_like(self.data, dtype=np.float)
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
        """Draw a representation of the quadtree on Matplotlib Axes ax."""
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

    # unit test for quad tree
    epsilon = 0.1
    seed = 1
    # Data generation
    data_dimension = 128
    num_center = 30
    sparsity_sigma = 3
    population = 100000
    # data = generate_synthetic_dataset_uniform(data_dimension, population)
    dataset = Dataset(data_dimension, population, seed=0)
    data = dataset.generate_synthetic_dataset_gaussian(sparsity_sigma=10)
    Dataset.plot_heatmap(data, annotation=False, valfmt="{x:d}")
    # Query workload generation
    num_queries = 1000
    query_list = dataset.generate_queries(num_queries)
    # Compute answer of the query

    # Quad tree standard
    #data_private_quad_std, quad_tree = QuadTree_standard(data, epsilon, max_height=None, max_points=5, seed=seed)
    data_private_quad_uniform, quad_tree_uniform = QuadTree_uniform(data, epsilon, max_height=8,
                                                                    max_points=128,  max_cells=128, seed=seed)
    data_private_quad_geometric, quad_tree_geometric = QuadTree_geometric(data, epsilon, max_height=8,max_cells=128,
                                                                          max_points=128, seed=seed)

    #quad_tree_geometric.draw()
    #quad_tree_uniform.draw()
    #print(quad_tree_uniform.total_height)
    # plot_grids(ag_grids, ax)
    # plt.show()
    # print("Quad-std: {:.4f}".format(evaluation_mre(data, data_private_quad_std, query_list)))
    print("Quad-uniform: {:.4f}".format(evaluation_mre(data, data_private_quad_uniform, query_list)))
    print("Quad-geometric: {:.4f}".format(evaluation_mre(data, data_private_quad_geometric, query_list)))
