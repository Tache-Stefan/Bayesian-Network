import pandas as pd
import math

class BicScore:
    def __init__(self, data):
        """
        Initializes the BIC score calculator with the given data.
        """
        self.data = data
        self.n_samples = len(data)
        self.state_counts = {col: data[col].nunique() for col in data.columns}
        self.cache = {}
    
    def local_score(self, node, parents):
        """
        Computes the local BIC score for a given node and its parents.
        """
        cache_key = f"{node}_" + "_".join(sorted(parents))
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        q = 1 # number of parent configurations
        for p in parents:
            q *= self.state_counts[p]
        
        r = self.state_counts[node] # number of states of the node
        k = (r - 1) * q # number of parameters

        penalty = 0.5 * k * math.log(self.n_samples)

        if not parents:
            counts = self.data[node].value_counts()
            ll = 0
            for count in counts:
                if count > 0:
                    ll += count * math.log(count / self.n_samples)
        else:
            cols = [node] + parents
            counts = self.data.groupby(cols, observed=True).size().reset_index(name='count')
            parent_counts = self.data.groupby(parents, observed=True).size().reset_index(name='parent_count')

            merged = pd.merge(counts, parent_counts, on=parents)

            ll = 0
            for _, row in merged.iterrows():
                N_ijk = row['count'] # Node + Parents frequency
                N_ij = row['parent_count'] # Parents frequency

                if N_ijk > 0 and N_ij > 0:
                    ll += N_ijk * math.log(N_ijk / N_ij)
        
        final_score = ll - penalty
        self.cache[cache_key] = final_score
        return final_score

    def compute_total_score(self, graph):
        """
        Computes the total BIC score.
        """
        total_score = 0
        for node in graph.nodes:
            parents = graph.get_parents(node)
            total_score += self.local_score(node, parents)
        return total_score
