from src.graph import DAG

class HillClimbSearch:
    def __init__(self, nodes, scoring_method):
        """
        Initializes the hill climbing search.
        """
        self.nodes = nodes
        self.scoring_method = scoring_method
        self.best_graph = DAG(nodes)

    def fit(self, skeleton, max_iterations=100, max_in_degree=4):
        """
        Runs the hill climbing search algorithm.
        """

        current_score = self.scoring_method.compute_total_score(self.best_graph)

        iteration = 0
        while iteration < max_iterations:
            best_operation = None
            best_u, best_v = None, None
            max_score_improvement = 0

            for u in self.nodes:
                for v in self.nodes:
                    if u == v or (tuple(sorted((u, v))) not in skeleton):
                        continue

                    for operation in ["add", "remove", "reverse"]:
                        if operation == "add" and len(self.best_graph.get_parents(v)) >= max_in_degree:
                            continue
                        if operation == "reverse" and len(self.best_graph.get_parents(u)) >= max_in_degree:
                            continue

                        test_graph = self.best_graph.copy()
                        valid_move = False

                        if operation == "add":
                            valid_move = test_graph.add_edge(u, v)
                        elif operation == "remove":
                            valid_move = test_graph.remove_edge(u, v)
                        elif operation == "reverse":
                            valid_move = test_graph.reverse_edge(u, v)
                        
                        if valid_move:
                            new_score = self.scoring_method.compute_total_score(test_graph)
                            improvement = new_score - current_score 

                            if improvement > max_score_improvement:
                                max_score_improvement = improvement
                                best_operation = operation
                                best_u, best_v = u, v
            
            if max_score_improvement > 0:
                if best_operation == "add":
                    self.best_graph.add_edge(best_u, best_v)
                elif best_operation == "remove":
                    self.best_graph.remove_edge(best_u, best_v)
                elif best_operation == "reverse":
                    self.best_graph.reverse_edge(best_u, best_v)

                current_score += max_score_improvement
                iteration += 1
            else:
                break
        
        return self.best_graph
