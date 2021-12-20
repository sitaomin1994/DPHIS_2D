import numpy as np


def compute_single_query(data: np.ndarray, query: tuple):
    """
    Compute a result of a query
    """
    x0, x_size, y0, y_size = query

    return int(data[x0:x0 + x_size, y0:y0 + y_size].sum())


def compute_all_queries(data: np.ndarray, query_list: list):
    """
    Compute result for a bunch of query workload
    """
    result = np.empty(len(query_list))
    for i, query in enumerate(query_list):
        result[i] = compute_single_query(data, query)

    return result


def evaluation_mre(data: np.ndarray, data_private: np.ndarray,
                   query_list: list, delta=0.001):
    """
    Evaluation the private release data based on mean relative error
     - |q_pri(D) - q(D)|/max(q(D),delta)
     - delta is a smoothing factor to deal with the zero counts
    """
    smoothing_factor = data.size * delta
    #smoothing_factor = 10

    true_result = compute_all_queries(data, query_list)
    private_result = compute_all_queries(data_private, query_list)
    diff = true_result - private_result
    divident = true_result.copy()
    divident[divident < smoothing_factor] = smoothing_factor

    return np.linalg.norm(np.absolute(diff)/divident, 1) / true_result.size


def evaluation_mae(data: np.ndarray, data_private: np.ndarray,
                   query_list: list, delta=0.001):
    """
    Evaluation the private release data based on mean absolute error
     - |q_pri(D) - q(D)|
    """
    true_result = compute_all_queries(data, query_list)
    private_result = compute_all_queries(data_private, query_list)
    diff = true_result - private_result

    return np.linalg.norm(diff,1) / true_result.size
