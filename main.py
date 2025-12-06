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
    rand = random.randint(0,int(summary))
    s=0
    for i,j in result:
        s+=j
        if j>=rand:
            return i
    return result[-1][0]
def cycle(start,graph,pheromones_graph,alpha,beta):
    """
    Docstring for cycle
    
    :param start: Description
    :param graph: Description
    :param pheromones_graph: Description
    :param alpha: Description
    :param beta: Description
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
def ant_algorithm(graph:dict, n:int, alpha=1, beta=2) -> list:
    """
    Write path to ant using ant_algorithm
    :param lst: graph with weights
    :param n: amount of ants
    :return: list of paths to each ant
    """
    pheromones_graph=pheromones(graph)
    lst=list(graph.keys())
    cycle_list=[]
    start=random.choice(lst)
    cycle_list.append(start)
    for i in range(n):
        cycl=cycle(start,graph,pheromones_graph,alpha,beta)
        cycle_list.append(cycl)
    return cycle_list
if __name__=='__main__':
    import doctest
    doctest.testmod()
    print(ant_algorithm({0: [(1, 1), (2, 1), (3, 1), (4, 1)],\
1: [(0, 1), (2, 1), (3, 1), (4, 1)],2: [(0, 1), (1, 1), (3, 1), (4, 1)],\
3: [(0, 1), (1, 1), (2, 1), (4, 1)],4: [(0, 1), (1, 1), (2, 1), (3, 1)]},30))
