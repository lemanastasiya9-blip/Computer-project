"""Module ant algorithm - Drawing effect like pen on paper"""
import random
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
def read_graph(filepath:str)->dict:
    """
    Reads an undirected graph from a file.

Parameters:
    filepath (str): path to the file (the first column contains the first vertex of the edge,
                    the second column — the second vertex, 
                    the third column — the weight of the edge)

Returns:
    dict: {vertex: [(neighbor_vertex, weight)]}
    """
    relation = {}
    with open(filepath, 'r', encoding='utf-8') as file1:
        for line in file1:
            if line:
                line = line.strip().split(',')
                relation.setdefault(int(line[1]), []).append((int(line[0]),float(line[2])))
                relation.setdefault(int(line[0]), []).append((int(line[1]),float(line[2])))
    return relation
def pheromones(graph: dict) -> dict:
    """Initialize pheromones on edges"""
    result = {}
    for node in graph.keys():
        for neighbor, weight in graph[node]:
            edge = tuple(sorted((node, neighbor)))
            if edge not in result:
                result[edge] = 0.0
    print(f"Initialized {len(result)} edges with pheromones:")
    for edge in sorted(result.keys()):
        print(f"  {edge}")
    return result
def choose_edge(current_edge: int, visited_edge, graph: dict,
                alpha: float, beta: float, pheromones_graph: dict) -> int:
    """Find next edge to visit"""
    result = []
    for i, j in graph[current_edge]:
        if i not in visited_edge:
            if (current_edge, i) in pheromones_graph:
                x = pheromones_graph[(current_edge, i)] + 0.1
            else:
                x = pheromones_graph[(i, current_edge)] + 0.1
            y = 1 / j
            probability = (x ** alpha) * (y ** beta)
            result.append((i, probability))
    if not result:
        return None
    summary = sum(i[1] for i in result)
    random_sum = random.random() * summary
    s = 0
    for i, j in result:
        s += j
        if s >= random_sum:
            return i
    return result[-1][0]
def cycle(start: int, graph: dict, pheromones_graph: dict,
          alpha: float, beta: float) -> list:
    """Find cycle"""
    visits = {start}
    path = [start]
    current_edge = start
    while len(path) != len(graph):
        next_edge = choose_edge(current_edge, visits, graph, alpha, beta, pheromones_graph)
        if next_edge is None:
            return None
        visits.add(next_edge)
        current_edge = next_edge
        path.append(next_edge)
    path.append(start)
    return path
def get_weight(graph, u, v):
    """Get weight of edge between u and v"""
    for to, w in graph[u]:
        if to == v:
            return w
    return float('inf')
def ant_algorithm(graph: dict, n: int, alpha=0.7, beta=0.3, evaporation=0.5) -> list:
    """
    Write path to ant using ant_algorithm
    :param graph: graph with weights
    :param n: amount of ants
    :return: list of paths to each ant
    """
    pheromones_graph = pheromones(graph)
    lst = list(graph.keys())
    best_path = None
    best_length = float('inf')
    start = random.choice(lst)
    cycle_list = [start]
    pheromone_history = []
    ant_paths_history = []
    for iteration in range(n):
        cycl = cycle(start, graph, pheromones_graph, alpha, beta)
        if cycl is not None:
            cycle_list.append(cycl)
            length = 0
            valid = True
            for i in range(len(cycl) - 1):
                w = next((w for v, w in graph[cycl[i]] if v == cycl[i+1]), None)
                if w is None:
                    valid = False
                    break
                length += w
            if not valid:
                continue
            if length < best_length:
                best_length = length
                best_path = cycl
        for edge in pheromones_graph:
            pheromones_graph[edge] *= (1 - evaporation)
        if best_path:
            for i in range(len(best_path)-1):
                edge = tuple(sorted((best_path[i], best_path[i+1])))
                pheromones_graph[edge] += 1.0 / best_length
        pheromone_history.append(pheromones_graph.copy())
        if cycl:
            ant_paths_history.append(cycl)
    return pheromone_history, ant_paths_history, [cycle_list, f'Hamiltonian cycle:{cycle_list[-1]}']

def ant_algorithm_step_by_step(graph: dict, n: int, alpha=0.7, beta=0.3,
                                evaporation=0.5):
    """
    Ant algorithm that records each step of each ant
    Returns detailed animation frames
    """
    pheromones_graph = pheromones(graph)
    lst = list(graph.keys())
    best_path = None
    best_length = float('inf')
    animation_frames = []
    animation_frames.append({
        'pheromones': pheromones_graph.copy(),
        'current_ant_position': None,
        'current_path': []
    })
    for iteration in range(n):
        start = random.choice(lst)
        visits = {start}
        path = [start]
        current_edge = start
        while len(path) != len(graph):
            next_edge = choose_edge(current_edge, visits, graph, alpha, beta, pheromones_graph)
            if next_edge is None:
                break
            visits.add(next_edge)
            path.append(next_edge)
            edge = tuple(sorted((current_edge, next_edge)))
            if edge in pheromones_graph:
                pheromones_graph[edge] += 10.0
            else:
                print(f"WARNING: Edge {edge} not found in pheromones!")
                pheromones_graph[edge] = 10.0
            animation_frames.append({
                'pheromones': pheromones_graph.copy(),
                'current_ant_position': next_edge,
                'current_path': path.copy()
            })
            current_edge = next_edge
        if len(path) == len(graph):
            path.append(start)
            edge = tuple(sorted((path[-2], start)))
            if edge in pheromones_graph:
                pheromones_graph[edge] += 10.0
            animation_frames.append({
                'pheromones': pheromones_graph.copy(),
                'current_ant_position': start,
                'current_path': path.copy()
            })
            length = 0
            for i in range(len(path) - 1):
                w = next((w for v, w in graph[path[i]] if v == path[i+1]), None)
                if w is not None:
                    length += w
            if length < best_length:
                best_length = length
                best_path = path
        for edge in pheromones_graph:
            pheromones_graph[edge] *= (1 - evaporation)
    return animation_frames
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
    parser.add_argument(
    "--path",
    type=str,
    default={
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
    },
    help="Шлях до файлу з графом (JSON)."
)
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
if __name__ == "__main__":
    main()

