import pandas as pd
from src.data_processor import DataProcessor
from src.mmpc import MMPC
from src.hill_climbing import HillClimbSearch
from src.scoring import BicScore
from src.probability import BayesianNetworkTables
from graphviz import Digraph

DataProcessor = DataProcessor("data/raw/heart.csv")
DataProcessor.pipeline("data/processed/heart_discrete.csv")
data = pd.read_csv("data/processed/heart_discrete.csv")

mmpc = MMPC(data, alpha=0.05)
skeleton = mmpc.get_skeleton()

hc = HillClimbSearch(list(data.columns), BicScore(data))
final_dag = hc.fit(skeleton=skeleton)

dot = Digraph(comment="Bayesian Network")
for node in final_dag.nodes:
    dot.node(node)

for u, v in final_dag.get_edges():
    dot.edge(u, v)

dot.render("output/bayes_net", format="png", cleanup=True)

bn_tables = BayesianNetworkTables(data, final_dag)
bn_tables.output_tables()

print("Bayesian Network structure and CPTs have been generated and saved to the output directory.")
