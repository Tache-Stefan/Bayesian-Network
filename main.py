import pandas as pd
from src.data_processor import DataProcessor
from src.mmpc import MMPC
from src.hill_climbing import HillClimbSearch
from src.scoring import BicScore
from src.probability import BayesianNetworkTables
from src.inference import BayesianNetworkInference
from graphviz import Digraph

# DataProcessor = DataProcessor("data/raw/heart.csv")
# DataProcessor.pipeline("data/processed/heart_discrete.csv")
data = pd.read_csv("data/processed/heart_discrete.csv")

mmpc = MMPC(data)
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

inference = BayesianNetworkInference(bn_tables)
inference.query("Heart Disease", value=1, evidence={"Age": "45-55", "Cholesterol Level": "High"})
inference.query("Heart Disease", value=1, evidence={"Age": "65+"})
inference.query("Heart Disease")

print("Inference results have been saved to output/inference.txt")
