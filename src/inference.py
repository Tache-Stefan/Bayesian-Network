import itertools
import os
import ast

class BayesianNetworkInference:
    def __init__(self, bn_tables):
        self.bn_tables = bn_tables
        self.dag = bn_tables.dag
        self.cpts = bn_tables.cpts
        self._possible_values = {}  
        self._discover_values()
    
    def _discover_values(self):
        """Discovers all possible values for each node based on the CPTs."""
        for node in self.dag.nodes:
            values = set()
            
            if isinstance(self.cpts[node], dict):
                if self.dag.get_parents(node):
                    for value_dist in self.cpts[node].values():
                        values.update(value_dist.keys())
                else:
                    values.update(self.cpts[node].keys())
            
            self._possible_values[node] = sorted(list(values))
    
    def _check_cache(self, query_string, file_path="output/inference.txt"):
        if not os.path.exists(file_path):
            return None
            
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(query_string + " = "):
                    result_text = line.split(" = ", 1)[1].strip()
                    try:
                        return ast.literal_eval(result_text)
                    except (ValueError, SyntaxError):
                        return None
        return None

    def _save_to_cache(self, query_string, result, file_path="output/inference.txt"):
        folder = os.path.dirname(file_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
            
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"{query_string} = {result}\n")
    
    def query(self, query_node, value=None, evidence=None):
        """
        Calculates P(query_node=value | evidence)
        If value is None, returns the complete distribution.
        """
        if evidence is None:
            evidence = {}
            
        if query_node not in self.dag.nodes:
            raise ValueError(f"Node {query_node} not in graph")
        
        sorted_evidence = sorted(evidence.items())
        ev_str = ", ".join([f"{k}={v}" for k, v in sorted_evidence]) if evidence else "None"
        val_str = f"={value}" if value is not None else " (distribution)"
        query_string = f"P({query_node}{val_str} | Evidence: {ev_str})"

        cached_result = self._check_cache(query_string)
        if cached_result is not None:
            print(f"Result for {query_string} found in cache.")
            return cached_result
            
        if value is None:
            distribution = {}
            total_prob = 0.0
            
            for v in self._possible_values[query_node]:
                prob = self._compute_marginal(query_node, v, evidence)
                distribution[v] = prob
                total_prob += prob
                
            if total_prob > 0:
                for k in distribution:
                    distribution[k] /= total_prob
            res = distribution
        else:
            # If a specific value is requested: P(Q=q | E) = P(Q=q, E) / P(E)
            prob_joint = self._compute_marginal(query_node, value, evidence)
            prob_evidence = self._compute_marginal(None, None, evidence)
            
            if prob_evidence > 0:
                res = prob_joint / prob_evidence
            else:
                res = 0.0
        
        self._save_to_cache(query_string, res)

        return res

    def _compute_marginal(self, query_node, query_value, evidence):
        """
        Sums the probabilities of all possible "worlds" that 
        satisfy the evidence (evidence) and the query (query_node).
        """
        constraints = evidence.copy()
        if query_node is not None:
            constraints[query_node] = query_value
            
        unconstrained_nodes = [n for n in self.dag.nodes if n not in constraints]
        
        if not unconstrained_nodes:
            return self._get_joint_prob(constraints)
            
        total_prob = 0.0
        
        unconstrained_values = [self._possible_values[n] for n in unconstrained_nodes]
        for combo in itertools.product(*unconstrained_values):
            state = constraints.copy()
            for i, node in enumerate(unconstrained_nodes):
                state[node] = combo[i]
            
            total_prob += self._get_joint_prob(state)
            
        return total_prob

    def _get_joint_prob(self, state):
        """
        Calculates the probability of a complete scenario,
        by multiplying the individual probabilities of all nodes.
        """
        prob = 1.0
        for node in self.dag.nodes:
            val = state[node]
            parents = self.dag.get_parents(node)
            
            if not parents:
                p = self.cpts[node].get(val, 0.0)
            else:
                parent_key = tuple(state[p] for p in parents)
                
                if parent_key not in self.cpts[node]:
                    return 0.0
                if val not in self.cpts[node][parent_key]:
                    return 0.0
                    
                p = self.cpts[node][parent_key][val]
                
            prob *= p
            
            # Optimization
            if prob == 0.0:
                break
                
        return prob
