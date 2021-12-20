import numpy as np
import random as random
from numpy.random import default_rng
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import rotate
import math

########################################################################################################################
# Random Sample helper
########################################################################################################################
def generate_random_samples(population, distri_name, loc, sigma, rng):
    # initialize random generator
    # print(distri_name, loc, sigma, low, u_length)

    # random choose a distribution for x and y
    if distri_name == 'normal':
        samples = rng.normal(loc, sigma / 2.5, population)
    elif distri_name == 'uniform':
        low = loc - sigma
        high = loc + sigma
        samples = rng.uniform(low, high, population)
    else:
        raise ValueError('Algorithm Error')

    return samples


########################################################################################################################
# Dataset Class
########################################################################################################################
class Dataset:

    def __init__(self, dimension, population: int = 1000, seed: int = 0):

        self.dimension = dimension
        self.population = population
        self.seed = seed
        self.rng = default_rng(seed)

    def generate_random_datasets(self, sigma_list, shape_list, population_list, num_datasets):

        datasets = []
        x_rng = self.rng
        x_sigma = x_rng.choice(sigma_list, num_datasets)
        x_loc = x_rng.integers(x_sigma, self.dimension - x_sigma + 1, num_datasets)

        self.rng.shuffle(sigma_list)

        y_rng = self.rng
        y_sigma = y_rng.choice(sigma_list, num_datasets)
        y_loc = y_rng.integers(y_sigma, self.dimension - y_sigma + 1, num_datasets)

        for i in range(num_datasets):

            data = np.zeros((self.dimension, self.dimension), dtype=np.int)
            shape = shape_list[i]
            population = population_list[i]

            if shape == 'circle':
                x_samples = list(
                    generate_random_samples(population, 'normal', x_loc[i], x_sigma[i], rng=self.rng))
                y_samples = list(
                    generate_random_samples(population, 'normal', y_loc[i], x_sigma[i], rng=self.rng))
            elif shape == 'ellipse':
                x_samples = list(
                    generate_random_samples(population, 'normal', x_loc[i], x_sigma[i], rng=self.rng))
                y_samples = list(
                    generate_random_samples(population, 'normal', y_loc[i], y_sigma[i], rng=self.rng))
            elif shape == 'square':
                x_samples = list(
                    generate_random_samples(population, 'uniform', x_loc[i], x_sigma[i], rng=self.rng))
                y_samples = list(
                    generate_random_samples(population, 'uniform', y_loc[i], x_sigma[i], rng=self.rng))
            elif shape == 'rectangle':
                x_samples = list(
                    generate_random_samples(population, 'normal', x_loc[i], x_sigma[i], rng=self.rng))
                y_samples = list(
                    generate_random_samples(population, 'uniform', y_loc[i], y_sigma[i], rng=self.rng))
            elif shape == 'strip':
                num_strip = 3
                shifts = self.rng.integers(0, self.dimension, num_strip)
                x_samples, y_samples = [], []
                for j in range(num_strip):
                    shift = shifts[j]
                    x_sample = list(
                        generate_random_samples(int(population / num_strip), 'normal', int(x_loc[i] + shift), x_sigma[i],
                                                rng=self.rng))
                    y_sample = list(
                        generate_random_samples(int(population / num_strip), 'uniform', self.dimension // 2,
                                                self.dimension //2, rng=self.rng))
                    x_samples.extend(x_sample)
                    y_samples.extend(y_sample)
            elif shape == 'crossing':
                x_samples = list(
                    generate_random_samples(population // 2, 'normal', x_loc[i],
                                            x_sigma[i], rng=self.rng))
                y_samples = list(
                    generate_random_samples(population // 2, 'uniform', self.dimension // 2,
                                            self.dimension // 2, rng=self.rng))

                x_samples.extend(list(
                    generate_random_samples(population // 2, 'uniform', self.dimension // 2,
                                            self.dimension // 2, rng=self.rng)))
                y_samples.extend(list(
                    generate_random_samples(population // 2, 'uniform', y_loc[i], x_sigma[i], rng=self.rng)))
            else:
                raise ValueError('Shape not support.')

            for x, y in zip(x_samples, y_samples):
                # (x,y)
                x, y = int(x), int(y)
                if x < 0: x = 0
                if x >= self.dimension: x = self.dimension - 1
                if y < 0: y = 0
                if y >= self.dimension: y = self.dimension - 1
                # if x < 0: x = self.dimension - abs(x) % self.dimension
                # if x >= self.dimension: x = x % self.dimension
                # if y < 0: y = self.dimension - abs(y) % self.dimension
                # if y >= self.dimension: y = y % self.dimension

                data[x, y] = data[x, y] + 1

            datasets.append(data)

        return datasets

    def generate_synthetic_dataset_gaussian(self, num_center: int = 1, sparsity_sigma: int = 10, fixed=False,
                                            location=None):

        # set random seed
        random.seed(self.seed)
        rng = default_rng(self.seed)
        # generate zero-like 2D array to store the data
        data = np.zeros((self.dimension, self.dimension), dtype=np.int)
        num_data_per_cluster = int(self.population)
        print(num_data_per_cluster)

        # generate n random centers
        if fixed == False:
            cluster_centers = [(random.randint(0, self.dimension), random.randint(0, self.dimension)) for _ in
                               range(num_center)]
        else:
            if location:
                cluster_centers = location
            else:
                cluster_centers = [(self.dimension // 2, self.dimension // 2)]

        # for each center, generator and clusters based on guassian distribution
        for center in cluster_centers:
            x_indices = list(rng.normal(center[0], sparsity_sigma, num_data_per_cluster))
            y_indices = list(rng.normal(center[1], sparsity_sigma, num_data_per_cluster))
            for x, y in zip(x_indices, y_indices):
                x, y = int(x), int(y)
                if x < 0: x = self.dimension - abs(x) % self.dimension
                if x >= self.dimension: x = x % self.dimension
                if y < 0: y = self.dimension - abs(y) % self.dimension
                if y >= self.dimension: y = y % self.dimension

                data[x, y] = data[x, y] + 1

        # print(data.sum())
        # print(data.mean())
        # print(cluster_centers)
        # print(data.shape)
        return data

    def generate_dataset_1D_gaussian(self, num_center: int = 1, sparsity_sigma: int = 10, axis=0, fixed=False,
                                     location=None):

        # set random seed
        random.seed(self.seed)
        rng = default_rng(self.seed)
        # generate zero-like 2D array to store the data
        data = np.zeros((self.dimension, self.dimension), dtype=np.int)
        num_data_per_cluster = int(self.population)
        print(num_data_per_cluster)

        # generate n random centers
        if fixed == False:
            cluster_centers = [random.randint(0, self.dimension) for _ in
                               range(num_center)]
        else:
            if location:
                cluster_centers = location
            else:
                cluster_centers = [self.dimension // 2]

        # for each center, generator and clusters based on guassian distribution
        for center in cluster_centers:
            if axis == 0:
                x_indices = list(rng.normal(center, sparsity_sigma, num_data_per_cluster))
                y_indices = list(rng.uniform(0, self.dimension, num_data_per_cluster))
            else:
                x_indices = list(rng.uniform(0, self.dimension, num_data_per_cluster))
                y_indices = list(rng.normal(center, sparsity_sigma, num_data_per_cluster))

            for x, y in zip(x_indices, y_indices):
                x, y = int(x), int(y)
                if x < 0: x = self.dimension - abs(x) % self.dimension
                if x >= self.dimension: x = x % self.dimension
                if y < 0: y = self.dimension - abs(y) % self.dimension
                if y >= self.dimension: y = y % self.dimension

                data[x, y] = data[x, y] + 1

        # print(data.sum())
        # print(data.mean())
        # print(cluster_centers)
        # print(data.shape)
        data = rotate(data, angle=45)
        data[data < 0] = 0
        return data

    # def generate_synthetic_dataset_uniform(self):
    #
    #     # set random seed
    #     random.seed(self.seed)
    #     rng = default_rng(self.seed)
    #
    #     # generate zero-like 2D array to store the data
    #     data = np.zeros((self.dimension, self.dimension), dtype=np.int)
    #     num_data_per_cluster = int(population)
    #     print(num_data_per_cluster)
    #
    #     # generate n random centers
    #     x_indices = list(rng.uniform(0, self.dimension, population))
    #     y_indices = list(rng.uniform(0, self.dimension, population))
    #
    #     for x, y in zip(x_indices, y_indices):
    #         x, y = int(x), int(y)
    #         data[x, y] = data[x, y] + 1
    #
    #     # for i in range(data.shape[0]):
    #     #     print(data[i,])
    #
    #     print(data.sum())
    #     print(data.mean())
    #     print(data.shape)
    #     return data

    def generate_queries(self, query_num: int = 2000, query_range_list=None):
        """
        Generate a list of random query, each query covers a range of space in the 2D data space
        """
        data_dimension = self.dimension
        if query_range_list is None:
            query_range_list = [0.02, 0.06, 0.10]

        # range_limit_lb = round(data_dimension * 0.7)
        range_limit_lb = int(data_dimension * 0.08)
        if range_limit_lb == 0:
            range_limit_lb = 1
        # range_limit_ub = int(data_dimension * 0.8)
        range_limit_ub = int(data_dimension * 0.9)

        query_list = []
        query_size_x = self.rng.integers(range_limit_lb, range_limit_ub, query_num)
        #query_size_y = self.rng.integers(range_limit_lb, range_limit_ub, query_num)
        query_ratio = self.rng.random(query_num)*0.95 + 0.01
        query_size_y = query_size_x*query_ratio
        #print(query_size_y)
        query_x = self.rng.integers(0, data_dimension - query_size_x - 1, query_num)
        query_y = self.rng.integers(0, data_dimension - query_size_y - 1, query_num)

        for i in range(query_num):
            query_list.append((query_x[i], query_size_x[i], query_y[i], int(query_size_y[i])))

        return query_list

    def data_stats(self, datasets):

        data_occupies = np.empty(len(datasets))
        data_variances = np.empty(len(datasets))
        for i, data in enumerate(datasets):
            data_occupies[i] = np.count_nonzero(data)/self.dimension**2
            data_variances[i] = np.std(data[data>0])

        print("Data stats: ")
        print("- occupies {:.3f} + {:.4f}".format(data_occupies.mean(), data_occupies.std()))
        print("- variance {:.3f} + {:.4f}".format(data_variances.mean(), data_variances.std()))

    def query_stats(self, query_list):

        query_sizes = []
        query_ratios = []
        for query in query_list:
            h, w = query[1], query[3]
            query_size = h * w
            query_wh_ratio = h / w if h <= w else w / h

            query_sizes.append(query_size/(self.dimension**2))
            query_ratios.append(query_wh_ratio)

        print('Size Ratio')
        N = len(query_sizes)
        size_mean = sum(query_sizes) / N
        size_std = math.sqrt(sum((size - size_mean)**2 for size in query_sizes)/ N)
        ratio_mean = sum(query_ratios) / N
        ratio_std = math.sqrt(sum((ratio - ratio_mean)**2 for ratio in query_ratios)/N)
        print("Query: size {:.3f} + {:.4f} ratio {:.3f} + {:.3f}".format(size_mean, size_std, ratio_mean, ratio_std))

    @staticmethod
    def plot_data(data):
        DPI = 144
        data_dimension = data.shape[0]

        # heatmap
        plt.imshow(data, interpolation='none', vmin=0, vmax=data[data > 0].mean())
        plt.show()

        # scatter plot
        # fig = plt.figure(figsize=(700 / DPI, 700 / DPI), dpi=DPI)
        # ax = plt.subplot()
        # ax.set_xlim(0, data_dimension)
        # ax.set_ylim(0, data_dimension)
        #
        # points_x, points_y, counts = [], [], []
        # for i in range(data.shape[0]):
        #     for j in range(data.shape[1]):
        #         if data[i, j] > 0:
        #             points_x.append(i)
        #             points_y.append(j)
        #             counts.append(data[i, j])
        #
        # ax.scatter(points_x, points_y, s=10, c=counts)
        # return ax

    @staticmethod
    def plot_heatmap(data, annotation=False, valfmt="{x:.1f}"):
        # data_noisy = data_noisy / data_noisy.sum()

        fig, ax = plt.subplots()
        im = ax.imshow(data, interpolation='none', vmin=0, vmax=data[data > 0].mean())
        cbar = ax.figure.colorbar(im, ax=ax)

        if annotation:
            threshold = im.norm(data.max()) * 0.1
            textcolors = ("white", "black")
            kw = dict(horizontalalignment="center",
                      verticalalignment="center")
            if isinstance(valfmt, str):
                valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                    text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)

        fig.tight_layout()
        plt.show()

    @staticmethod
    def plot_query(data, query: list, c='k', lw=1, **kwargs):
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

        ax.scatter(points_x, points_y, s=10, c=counts)

        for grid in query:
            x0, h, y0, w = grid
            x1, y1 = x0, y0
            x2, y2 = x0 + h, y0 + w
            ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], c=c, lw=lw, **kwargs)

        plt.show()


if __name__ == '__main__':
    epsilon = 0.1
    # Data generation
    data_dimension = 128
    seed = 1
    rng = default_rng(seed)

    # data = generate_synthetic_dataset_uniform(data_dimension, population)
    num_data = 10
    populations = [1000, 2500, 5000, 10000]
    shapes = ['circle']
    sigma = [0.05, 0.1, 0.15, 0.4]

    dispersion = [s * data_dimension for s in sigma]
    shape_list = rng.choice(shapes, num_data)
    population_list = rng.choice(populations, num_data)
    # dispersion = [2, 4, 8, 10, 20, 40, 60, 80, 100, 120, 160, 200]
    dataset = Dataset(data_dimension, seed=seed)
    datasets = dataset.generate_random_datasets(dispersion, shape_list, population_list, num_data)
    # print('a', len(datasets))
    for data in datasets:
        dataset.plot_heatmap(data)

    # Query workload generation
    num_queries = 1000
    query_list = dataset.generate_queries(num_queries)
    print(query_list)
