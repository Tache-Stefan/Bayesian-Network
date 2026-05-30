class DAG:
    def __init__(self, nodes):
        """
        Initializes a Directed Acyclic Graph (DAG) with the given nodes (columns names).
        """
        self.nodes = nodes
        self.adj_list = {node: [] for node in nodes}
    
    def copy(self):
        """
        Creates a copy of the graph.
        """
        new_graph = DAG(self.nodes)
        new_graph.adj_list = {node: list(children) for node, children in self.adj_list.items()}
        return new_graph
    
    def add_edge(self, from_node, to_node):
        """
        Adds a directed edge from 'from_node' to 'to_node'.
        Return True if no cycles, False otherwise.
        """
        if from_node == to_node or to_node in self.adj_list[from_node]:
            return False
        
        self.adj_list[from_node].append(to_node)
        if self.has_cycle():
            self.adj_list[from_node].remove(to_node)
            return False

        return True
    
    def remove_edge(self, from_node, to_node):
        """
        Removes the directed edge from 'from_node' to 'to_node'.
        """
        if to_node in self.adj_list[from_node]:
            self.adj_list[from_node].remove(to_node)
            return True
        return False
    
    def reverse_edge(self, from_node, to_node):
        """
        Reverses the directed edge from 'from_node' to 'to_node'.
        Return True if no cycles, False otherwise.
        """
        if to_node not in self.adj_list[from_node]:
            return False
        
        self.adj_list[from_node].remove(to_node)
        self.adj_list[to_node].append(from_node)

        if self.has_cycle():
            self.adj_list[to_node].remove(from_node)
            self.adj_list[from_node].append(to_node)
            return False
    
        return True
    
    def get_parents(self, node):
        """
        Returns a list of parent nodes for the given node.
        """
        return [parent for parent in self.nodes if node in self.adj_list[parent]]
    
    def get_edges(self):
        """
        Returns a list of all edges in the graph.
        """
        edges = []
        for node, children in self.adj_list.items():
            for child in children:
                edges.append((node, child))
        return edges
    
    def has_cycle(self):
        """
        Checks if the graph has a cycle using Depth-First Search (DFS) with 3 states.
        States:
        0: unvisited
        1: visiting
        2: visited        
        """
        state = {node: 0 for node in self.nodes}

        def dfs(node):
            state[node] = 1
            for neighbor in self.adj_list[node]:
                if state[neighbor] == 1:
                    return True
                if state[neighbor] == 0 and dfs(neighbor):
                    return True
            state[node] = 2
            return False
        
        for node in self.nodes:
            if state[node] == 0 and dfs(node):
                return True
        
        return False
