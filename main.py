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
def get_weight(graph, u, v):
    """Get weight of edge between u and v"""
    for to, w in graph[u]:
        if to == v:
            return w
    return float('inf')
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
def visualize_ant_algorithm(graph, steps=15, alpha=0.7, beta=0.3, evaporation=0.5):
    """Visualize ants drawing the graph step by step"""
    frames = ant_algorithm_step_by_step(graph, steps, alpha, beta, evaporation)
    g = nx.Graph()
    for node, edges in graph.items():
        for v, w in edges:
            g.add_edge(node, v, weight=w)
    pos = nx.spring_layout(g, seed=42)
    fig, ax = plt.subplots(figsize=(10, 8))
    all_values = []
    for frame_data in frames:
        all_values.extend(frame_data['pheromones'].values())
    global_max = max(all_values) if all_values else 1
    def update(frame_idx):
        ax.clear()
        frame_data = frames[frame_idx]
        pher = frame_data['pheromones']
        nx.draw_networkx_nodes(
            g, pos,
            node_color="#ffcc66",
            node_size=1000,
            ax=ax
        )
        nx.draw_networkx_labels(
            g, pos,
            font_size=14,
            font_weight='bold',
            ax=ax
        )
        for u, v in g.edges():
            edge = tuple(sorted((u, v)))
            pheromone_value = pher[edge]
            if pheromone_value < 0.1:
                x_values = [pos[u][0], pos[v][0]]
                y_values = [pos[u][1], pos[v][1]]
                ax.plot(x_values, y_values, color=(0.95, 0.95, 0.95),
                       linewidth=0.3, linestyle='--', zorder=0, alpha=0.5)
            else:
                normalized = pheromone_value / global_max if global_max > 0 else 0
                width = 0.5 + 7.5 * normalized
                intensity = 0.95 - normalized * 0.85
                color = (intensity, intensity, intensity)
                x_values = [pos[u][0], pos[v][0]]
                y_values = [pos[u][1], pos[v][1]]
                ax.plot(x_values, y_values, color=color, linewidth=width, zorder=1)
        if frame_data['current_ant_position'] is not None:
            ant_pos = pos[frame_data['current_ant_position']]
            ax.plot(ant_pos[0], ant_pos[1], 'ro', markersize=15, zorder=5)
        ax.set_title(f"Ants Drawing Paths - Frame {frame_idx+1}/{len(frames)}",
                    fontsize=14, fontweight='bold')
        ax.set_xlim([min(p[0] for p in pos.values())-0.2, max(p[0] for p in pos.values())+0.2])
        ax.set_ylim([min(p[1] for p in pos.values())-0.2, max(p[1] for p in pos.values())+0.2])
        ax.axis('off')
    ani = FuncAnimation(fig, update, frames=len(frames),
                       interval=200, repeat=True)
    plt.tight_layout()
    plt.show()
def main():
    parser = argparse.ArgumentParser(description="Ant algorithm CLI tool")
    parser.add_argument("--ants", type=int, default=30,
                        help="Кількість мурах (ітерацій)")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Коефіцієнт alpha")
    parser.add_argument("--beta", type=float, default=0.3,
                        help="Коефіцієнт beta")
    parser.add_argument("--evaporation", type=float, default=0.5,
                        help="Коефіцієнт випаровування феромонів")
    parser.add_argument("--animate", action="store_true",
                        help="Показати анімацію знайденого циклу")
    args = parser.parse_args()
    graph = {
        0: [(1,3), (9,3), (2,6), (4,9), (6,2)],
        1: [(0,3), (2,3), (3,7), (7,9)],
        2: [(1,3), (3,3), (0,6), (5,5), (8,2)],
        3: [(2,3), (4,4), (1,7), (6,6)],
        4: [(3,4), (5,3), (0,9), (7,5)],
        5: [(4,3), (6,3), (2,5), (8,6)],
        6: [(5,3), (7,4), (0,2), (3,6)],
        7: [(6,4), (8,3), (4,5), (1,9)],
        8: [(7,3), (9,3), (2,2), (5,6)],
        9: [(8,3), (0,3), (3,8)]
    }
    print("Running ant algorithm...")
    result = ant_algorithm(graph, 30)
    print(result[2])
    print("\nStarting visualization...")
    print("Watch as ants draw the graph step by step!")
    if args.animate:
        visualize_ant_algorithm(
            graph,
            steps=15,
            alpha=0.7,
            beta=0.3,
            evaporation=0.5
        )
if __name__=='__main__':
    import doctest
    doctest.testmod()
    main()
