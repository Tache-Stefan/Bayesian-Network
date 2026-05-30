import pandas as pd
from src.data_processor import DataProcessor
from src.mmpc import MMPC
from src.hill_climbing import HillClimbSearch
from src.scoring import BicScore
from src.probability import ProbabilityCalculatorWithDAG
from graphviz import Digraph

DataProcessor = DataProcessor("data/raw/heart.csv")
DataProcessor.load_data()
DataProcessor.rename_columns()
DataProcessor.discretize_data()
DataProcessor.save_processed_data("data/processed/heart_discrete.csv")
data = pd.read_csv("data/processed/heart_discrete.csv")

mmpc = MMPC(data, alpha=0.05)
skeleton = mmpc.get_skeleton()

scorer = BicScore(data)
hc = HillClimbSearch(list(data.columns), scorer)
final_dag = hc.fit(skeleton=skeleton)
target_node = "Heart Disease"

dot = Digraph(comment="Bayesian Network")
for node in final_dag.nodes:
    dot.node(node)

for u, v in final_dag.get_edges():
    dot.edge(u, v)

dot.render("bayes_net", format="png", cleanup=True)
print("Saved bayes_net.png")

prob_calc = ProbabilityCalculatorWithDAG(data, final_dag)

heart_disease_prob = prob_calc.query(target_node, 1)
print(f"P(Heart Disease = 1): {heart_disease_prob:.3f}")

age_evidence = {"Age": "65+"}
prob_with_age = prob_calc.query(target_node, 1, evidence=age_evidence)
print(f"P(Heart Disease = 1 | Age = 65+): {prob_with_age:.3f}")

evidence = {"Exercise Induced Angina": 0}
prob_with_evidence = prob_calc.query(target_node, 1, evidence=evidence)
print(f"P(Heart Disease = 1 | Exercise Induced Angina = 0): {prob_with_evidence:.3f}")
