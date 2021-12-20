import matplotlib.pyplot as plt
import numpy as np
import json


def plot_sigma_error(algo_result, sigma_list, algo_list):
    fig, ax = plt.subplots(figsize = (10,6))
    x = np.arange(len(sigma_list))
    x_label = sigma_list

    print(algo_result)

    for algo in algo_result:
        if algo in algo_list:
            result_mean = []
            result_std = []
            for sigma, result in algo_result[algo].items():
                result_np = np.array(result)
                result_mean.append(result_np.mean())
                result_std.append(result_np.std())
            if algo in {'AG', 'htf'}:
                ax.plot(x, np.log10(result_mean), marker = 'D', label=algo)
            else:
                ax.plot(x, np.log10(result_mean), 'o', label=algo)
    ax.set_xticks(x)
    ax.set_yticks(np.arange(-2, 3))
    ax.set_yticklabels(10.0 ** np.arange(-2, 3))
    ax.set_xticklabels(x_label, rotation=0, fontsize=10)
    ax.set_xlabel("Sigma")
    ax.set_ylabel('MRE')

    plt.legend()
    plt.show()


def plot_box_error(algo_result_error, name_list, name_refs):
    fig, ax = plt.subplots(figsize = (10,6))
    data = []
    for algo_name in name_list:
        data.append(algo_result_error[algo_name])

    ax.boxplot(np.log10(np.array(data).T), '')
    ax.set_xticklabels(name_refs, rotation=45, fontsize=10)

    ax.set_yticks(np.arange(-2, 3))
    ax.set_yticklabels(10.0 ** np.arange(-2, 3))
    ax.set_ylabel('MRE')
    ax.set_title('')

    plt.tight_layout(pad=2, w_pad=1.0, h_pad=2.0)
    plt.show()


def plot_box_run_time(algo_result_run_time, name_list, name_refs):

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(name_list))
    data = []
    for algo_name in name_list:
        data.append(algo_result_run_time[algo_name])

    ax.bar(x, data)
    ax.set_xticks(x)
    ax.set_xticklabels(name_refs, rotation=45, fontsize=10)

    ax.set_ylabel('Time')
    ax.set_title('')

    plt.tight_layout(pad=2, w_pad=1.0, h_pad=2.0)
    plt.show()


def read_json(file_list, name_list):
    experiment_result = {}
    for file, name in zip(file_list, name_list):
        with open(file) as f:
            experiment_result[name] = json.load(f)
    return experiment_result


def concate_result(experiment_result):
    error = {}
    run_time = {}
    for k, v in experiment_result.items():
        for algo, algo_result in v['algos'].items():
            if not algo in error:
                error[algo] = algo_result['result']
            else:
                error[algo] = error[algo] + algo_result['result']

            if not algo in run_time:
                run_time[algo] = algo_result['run_time']
            else:
                run_time[algo] += algo_result['run_time']

    return error, run_time


def reformat_result(experiment_result):
    result = {}

    for sigma in experiment_result:
        for algo in experiment_result[sigma]['algos']:
            if algo not in result:
                result[algo] = {}

            result[algo][sigma] = experiment_result[sigma]['algos'][algo]['result']

    return result

if __name__ == '__main__':
    file_list = ['./result_30.json', './result_31.json', './result_32.json',
                 './result_33.json', './result_34.json']

    sigma_list = ['0.05', '0.1', '0.2', '0.35', '0.5']

    experiment_result = read_json(file_list, sigma_list)

    combined_result, concate_result_run_time = concate_result(experiment_result)
    reformat_result = reformat_result(experiment_result)

    algo_name_list = ['idt', 'uniform', 'AG', 'UG', 'Quad-uniform', 'Quad-geo', 'kd-uniform', 'kd-geo', 'kd-cell', 'kd-mean', 'htf']
    algo_name_ref = ['idt', 'uni', 'AG', 'UG', 'quad(uni)', 'quad(geo)', 'k-d(uni)', 'k-d(geo)', 'k-d(cell)', 'k-d(mean)', 'htf']

    print(combined_result)
    plot_box_error(combined_result, algo_name_list, algo_name_ref)
    # print(concate_result_run_time)
    # plot_box_run_time(concate_result_run_time, algo_name_list, algo_name_ref)
    plot_sigma_error(reformat_result, sigma_list, algo_name_list)