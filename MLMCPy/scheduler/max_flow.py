import numpy as np
import argparse
import collections


def bfs(capacity_graph, source, sink, parents):
    """
    Returns true if there is a path from source 's' to sink 't' in
    residual graph. Also fills parent[] to store the path
    """
    num_vertices = len(capacity_graph)
    visited = [False] * num_vertices

    queue = collections.deque()
    queue.append(source)
    visited[source] = True

    while queue:
        u = queue.popleft()

        # Get all adjacent vertices's of the dequeued vertex u
        # If an adjacent has not been visited, then mark it
        # visited and enqueue it
        for v, capacity in enumerate(capacity_graph[u]):
            if not visited[v] and capacity > 0:
                queue.append(v)
                visited[v] = True
                parents[v] = u

    # If we reached sink in BFS starting from source, then return
    # true, else false
    return visited[sink]


def edmonds_karp(capacity_graph, source, sink):
    """
    Returns the maximum flow from s to t in the given graph
    """
    # This array is filled by BFS and to store path
    num_vertices = len(capacity_graph)
    parents = [-1] * num_vertices

    max_flow = 0  # There is no flow initially

    # Augment the flow while there is path from source to sink
    while bfs(capacity_graph, source, sink, parents):

        # Find minimum residual capacity of the edges along the
        # path filled by BFS. Or we can say find the maximum flow
        # through the path found.
        path_flow = float("inf")
        s = sink
        while s != source:
            path_flow = min(path_flow, capacity_graph[parents[s]][s])
            s = parents[s]

        # Add path flow to overall flow
        max_flow += path_flow

        # Update residual capacities of the edges and reverse edges along the path
        v = sink
        while v != source:
            u = parents[v]
            capacity_graph[u][v] -= path_flow
            capacity_graph[v][u] += path_flow
            v = parents[v]

    return max_flow


def get_capacity_graph(job_counts, time_costs, num_workers, max_time):
    num_jobs = sum(job_counts)
    num_vertices = num_jobs + num_workers + 2
    graph = np.zeros((num_vertices, num_vertices))
    job_start = 1
    worker_start = 1 + num_jobs

    # 0 is the source vertex
    # the last one is the sink vertex
    for count, cost in zip(job_counts, time_costs):
        job_end_exclusive = job_start + count
        graph[0, job_start:job_end_exclusive] = cost
        graph[job_start:job_end_exclusive, worker_start:-1] = cost
        job_start = job_end_exclusive
    graph[worker_start:-1, -1] = max_time
    return graph


def get_first_worker_index_in_graph(num_jobs):
    return num_jobs + 1


def get_optimal_allocation(job_counts, time_costs, num_workers):
    num_jobs = sum(job_counts)
    m = num_jobs // num_workers + 1
    min_time = 0
    max_time = 0
    sink_index = 1 + num_jobs + num_workers
    total_time_cost = np.sum(np.array(job_counts) * np.array(time_costs))
    n = 0
    for count, cost in zip(job_counts[::-1], time_costs[::-1]):
        if n + count < m:
            max_time += cost * count
            n += count
        else:
            max_time += (m - n) * cost
            break

    # binary search
    while max_time > min_time:
        mid = (min_time + max_time) // 2
        capacity_graph = get_capacity_graph(
            job_counts=job_counts, time_costs=time_costs, num_workers=num_workers, max_time=mid
        )
        max_flow = edmonds_karp(capacity_graph, 0, sink_index)
        if max_flow != total_time_cost:
            min_time = mid + 1
        else:
            max_time = mid

    capacity_graph = get_capacity_graph(
        job_counts=job_counts, time_costs=time_costs, num_workers=num_workers, max_time=max_time
    )
    edmonds_karp(capacity_graph, 0, sink_index)
    print(capacity_graph)

    return max_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('max_time', type=int)
    args = parser.parse_args()

    job_counts = [1, 2, 2, 1]
    time_costs = [3, 4, 5, 12]
    num_workers = 3
    # total_time_cost = np.sum(np.array(job_counts) * np.array(time_costs))
    # graph = get_capacity_graph(
    #     job_counts=job_counts, time_costs=time_costs, num_workers=num_workers, max_time=args.max_time
    # )
    # print('=> capacity graph:')
    # print(graph)
    # max_flow = edmonds_karp(graph, 0, 10)
    # print(f'total time cost: {total_time_cost}\n'
    #       f'max flow: {max_flow}')

    print(get_optimal_allocation(job_counts, time_costs, num_workers))
