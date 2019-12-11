import numpy as np


def get_job_allocation(job_counts, costs, num_workers):
    """
    :param job_counts: the number of times each simulation model should be run.
    :param costs: the estimated time each model takes per run, assumed to be in increasing order.
    :param int num_workers: number of workers to run the simulations.
    """
    assert len(job_counts) == len(costs)


def get_job_allocation_heuristically(job_counts, costs, num_workers):
    """
    :param job_counts: the number of times each simulation model should be run.
    :param costs: the estimated time each model takes per run, assumed to be in increasing order.
    :param int num_workers: number of workers to run the simulations.
    :return (allocation, total_costs) where
    allocation is an np.array where each row corresponds to a worker and each column is the number of
    jobs for the corresponding model assigned to the worker and
    total_costs is the estimated total time each worker will take to finish the assigned jobs
    """
    assert len(job_counts) == len(costs)
    num_models = len(costs)

    allocation = np.zeros((num_workers, num_models), dtype='int64')
    total_costs = np.zeros(num_workers, dtype='int64')

    for i, (count, cost) in enumerate(zip(job_counts[::-1], costs[::-1])):
        model_index = num_models - 1 - i
        for _ in range(count):
            min_cost_index = np.argmin(total_costs)
            allocation[min_cost_index, model_index] += 1
            total_costs[min_cost_index] += cost
    return allocation, total_costs


if __name__ == '__main__':
    allocation = get_job_allocation_heuristically([1, 2, 2, 1], [3, 4, 5, 11], 3)
