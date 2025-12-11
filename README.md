# Ant Colony Optimization — Hamiltonian Cycle Finder
This module implements an Ant Colony Optimization (ACO) algorithm for approximating Hamiltonian cycles in weighted graphs.
The algorithm imitates how real ants find efficient paths using pheromone trails, allowing it to search for short cycles in complex graphs.
### Features
* Core ACO algorithm:
  * pheromone initialization
  * probabilistic edge selection
  * pheromone evaporation & reinforcement
  * repeated ant simulations
* Searches for a Hamiltonian cycle with the shortest discovered weight.
* Command-line interface for custom parameters.
* Optional graph visualization with animation.
### How the Algorithm Works
1. Initialization
Each undirected edge is assigned an initial pheromone level: 1.0.
2. Ant movement
Each ant:
* starts at a random node
* moves to unvisited nodes
* chooses the next node with probability:
* continues until all nodes are visited
3. Cycle evaluation
The total weight of the path is calculated using graph edge weights.
4. Pheromone update
* Evaporation reduces pheromones globally
* Reinforcement increases pheromones on the best cycle found so far
5. Result
The algorithm returns:
* all constructed cycles
* the best Hamiltonian cycle
### Code Structure
* pheromones(graph)
  #### Creates a dictionary of pheromone values for every edge.
* choose_edge(current, visited, graph, alpha, beta, pheromones)
  #### Selects the next vertex based on pheromones and edge weights.
* cycle(start, graph, pheromones, alpha, beta)
  #### Builds a Hamiltonian-like cycle from a start node.
* ant_algorithm(graph, n, alpha, beta, evaporation)
  #### Runs the full ant algorithm for n ants.
  #### Returns:
  * all cycles generated
  * the best Hamiltonian cycle
* visualize_ant_algorithm(graph, steps=15, alpha=0.7, beta=0.3, evaporation=0.5)
  #### Produces a matplotlib animation showing pheromone updates.
* main()
  #### CLI interface for running the algorithm and visualization.
### Teamwork
#### Lem Anastasiia
* ant algorithm
#### Sofiia Alania
* reading graph from file and ant algorithm
#### Kateryna Karalyus
* visualisation
#### Sofiia Tsiuk
* visualization
#### Polina Zuzuk
* presentation and documentation 


