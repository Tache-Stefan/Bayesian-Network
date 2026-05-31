import pandas as pd
import os

class BayesianNetworkTables:
    def __init__(self, data, dag):
        self.data = data
        self.dag = dag
        self.cpts = {}
        self._build_cpts()
        
    def _build_cpts(self):
        """Calculates the CPTs for each node."""
        for node in self.dag.nodes:
            parents = self.dag.get_parents(node)
            
            if not parents:
                self.cpts[node] = self.data[node].value_counts(normalize=True).sort_index().to_dict()
            else:
                self.cpts[node] = {}
                
                for parent_combo, group in self.data.groupby(parents):
                    # Ensure key is tuple
                    parent_key = parent_combo if isinstance(parent_combo, tuple) else (parent_combo,)
                    
                    counts = group[node].value_counts(normalize=True).sort_index()
                    self.cpts[node][parent_key] = counts.to_dict()

    def output_tables(self):
        """Outputs the CPTs and writes to output/CPT.txt."""
        os.makedirs('output', exist_ok=True)
        
        with open('output/CPT.txt', 'w') as f:
            for node in self.dag.nodes:
                f.write(f"\n" + "="*50 + "\n")
                f.write(f" CPT FOR THE NODE: '{node}'\n")
                f.write("="*50 + "\n")
                
                parents = self.dag.get_parents(node)
                
                if not parents:
                    df_marginal = pd.DataFrame(list(self.cpts[node].items()), columns=[node, 'Probability'])
                    f.write(df_marginal.to_string(index=False) + "\n")
                else:
                    table_data = []
                    
                    for parent_key, node_distribution in self.cpts[node].items():
                        for node_val, prob in node_distribution.items():
                            row = list(parent_key) + [node_val, prob]
                            table_data.append(row)
                    
                    columns = list(parents) + [node, 'Probability']
                    df_cpt = pd.DataFrame(table_data, columns=columns)
                    df_cpt = df_cpt.sort_values(by=list(parents) + [node])
                    f.write(df_cpt.to_string(index=False) + "\n")
    