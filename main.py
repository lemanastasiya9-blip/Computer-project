"""Module ant algorithm"""
import random
def pheromones(graph:dict) -> dict:
    """
    pheromones on edge
    :param graph: graph with edge
    :return: dictionar with pheromons on edge
    >>> pheromones({0: [(1,10), (2,20)], 1: [(0,10), (2,30)], 2: [(0,20), (1,30)]})
    {(0, 1): 1.0, (0, 2): 1.0, (1, 2): 1.0}
    """
    result = {}
    keys = graph.keys()
    for i in keys:
        for j in graph[i]:
            x = (i, j[0])
            edge = tuple(sorted(x))
            if edge not in result:
                result[edge] = 1.0
    return result
def choose_edge(curent_edge:int, visited_edge, graph:dict,\
    alpha:int, beta:int, pheromones_graph:dict) -> int:
    """
    Find next edge to visit
    :param curent_edge: edge in which we are now
    :param visited_edge: edges in which we were already
    :param graph: graph with edge
    :param alpha, beta coefficients 
    :param pheromones_graph: dict with pheromons
    :return: next edge to go
    """
    result=[]
    for i,j in graph[curent_edge]:
        if i not in visited_edge:
            if (curent_edge,i) in pheromones_graph:
                x=pheromones_graph[(curent_edge,i)]
            else:
                x=pheromones_graph[(i,curent_edge)]
            y=1/j
            probability=(x**alpha) * (y**beta)
            result.append((i,probability))
    if not result:
        return None
    summary=0
    for i in result:
        summary+=i[1]
    random_sum = random.random()*summary
    s=0
    for i,j in result:
        s+=j
        if j>=random_sum:
            return i
    return result[-1][0]
def cycle(start:int,graph:dict,pheromones_graph:dict,alpha:int,beta:int)->list:
    """
    Find cycle 
    :param start: start node
    :param graph: graph
    :param pheromones_graph: pheronomes on edges
    :param alpha: coefficient
    :param beta: coefficient
    :return list of path
    """
    visits={start}
    path=[start]
    curent_edge=start
    while len(path)!=len(graph):
        next_edge=choose_edge(curent_edge,visits,graph,alpha,beta,pheromones_graph)
        if next_edge is None:
            return None
        visits.add(next_edge)
        curent_edge=next_edge
        path.append(next_edge)
    path.append(start)
    return path
def ant_algorithm(graph:dict, n:int, alpha=0.7, beta=0.3, evaporation=0.5) -> list:
    """
    Write path to ant using ant_algorithm
    :param lst: graph with weights
    :param n: amount of ants
    :return: list of paths to each ant
    """
    pheromones_graph = pheromones(graph)
    lst = list(graph.keys())
    best_path = None
    best_length = float('inf')
    start = random.choice(lst)
    cycle_list = [start]
    for _ in range(n):
        cycl = cycle(start, graph, pheromones_graph, alpha, beta)
        if cycl is not None:
            cycle_list.append(cycl)
            length = sum(next(w for v,w in graph[cycl[i]] if v==cycl[i+1])\
                          for i in range(len(cycl)-1))
            if length < best_length:
                best_length = length
                best_path = cycl
        for edge in pheromones_graph:
            pheromones_graph[edge] *= (1 - evaporation)
        if best_path:
            for i in range(len(best_path)-1):
                edge = tuple(sorted((best_path[i], best_path[i+1])))
                pheromones_graph[edge] += 1.0 / best_length
    return [cycle_list, f'Hamiltonian cycle:{cycle_list[-1]}']
if __name__=='__main__':
    import doctest
    doctest.testmod()
    print(ant_algorithm({
    0: [(1, 1), (2, 4)],
    1: [(0, 1), (3, 5), (4, 1)],
    2: [(0, 4), (3, 1)],
    3: [(2, 1), (1, 5), (5, 1)],
    4: [(1, 1), (5, 1)],
    5: [(3, 1), (4, 1)]
},30))
