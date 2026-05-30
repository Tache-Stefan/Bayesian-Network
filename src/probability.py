class ProbabilityCalculator:
    def __init__(self, data, dag=None):
        """
        Initialize with data and optional DAG structure.
        """
        self.data = data
        self.dag = dag
        self.cpt = {}
        self._build_cpts()
    
    def _build_cpts(self):
        """Build conditional probability tables from data."""
        for col in self.data.columns:
            self.cpt[col] = self.data[col].value_counts(normalize=True).to_dict()
    
    def marginal_probability(self, variable, value):
        """
        Get P(variable = value).
        """
        if variable not in self.cpt:
            raise ValueError(f"Variable {variable} not found")
        
        return self.cpt[variable].get(value, 0.0)
    
    def conditional_probability(self, target_var, target_val, evidence):
        """
        Get P(target_var = target_val | evidence).
        """
        filtered_data = self.data.copy()
        
        for var, val in evidence.items():
            filtered_data = filtered_data[filtered_data[var] == val]
        
        if len(filtered_data) == 0:
            return 0.0
        
        matches = filtered_data[filtered_data[target_var] == target_val]
        
        return len(matches) / len(filtered_data)
    
    def query(self, target_var, target_val, evidence=None):
        """
        Simple query interface.
        """
        if evidence is None or len(evidence) == 0:
            return self.marginal_probability(target_var, target_val)
        else:
            return self.conditional_probability(target_var, target_val, evidence)
    
class ProbabilityCalculatorWithDAG(ProbabilityCalculator):
    def __init__(self, data, dag):
        super().__init__(data, dag)
    
    def _build_cpts(self):
        """Build conditional probability tables from data using DAG structure."""
        for node in self.dag.nodes:
            parents = self.dag.get_parents(node)
            
            if not parents:
                self.cpt[node] = self.data[node].value_counts(normalize=True).to_dict()
            else:
                self.cpt[node] = {}
                # Group by parents and calculate probabilities for each group
                for parent_combo, group in self.data.groupby(parents):
                    counts = group[node].value_counts()
                    total = counts.sum()
                    
                    # Handle single parent (parent_combo won't be a tuple)
                    if not isinstance(parent_combo, tuple):
                        key = (parent_combo,)
                    else:
                        key = parent_combo
                    
                    # Store conditional probabilities
                    self.cpt[node][key] = (counts / total).to_dict()
    
    def conditional_probability(self, target_var, target_val, evidence):
        """
        Get P(target_var = target_val | evidence) using only relevant parents.
        """
        parents = self.dag.get_parents(target_var)
        relevant_evidence = {p: evidence[p] for p in parents if p in evidence}
        
        if not relevant_evidence:
            # Use marginal probability if no relevant parents in evidence
            return self.data[self.data[target_var] == target_val].shape[0] / len(self.data)
        
        filtered_data = self.data.copy()
        for var, val in relevant_evidence.items():
            filtered_data = filtered_data[filtered_data[var] == val]
        
        if len(filtered_data) == 0:
            return 0.0
        
        matches = filtered_data[filtered_data[target_var] == target_val]
        return len(matches) / len(filtered_data)
    
    def query(self, target_var, target_val, evidence=None):
        """Simple query using the graph structure."""
        if evidence is None or len(evidence) == 0:
            return self.data[self.data[target_var] == target_val].shape[0] / len(self.data)
        else:
            return self.conditional_probability(target_var, target_val, evidence)
    