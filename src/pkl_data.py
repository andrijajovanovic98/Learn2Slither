import pickle

with open("models/qtable_10x10.pkl", "rb") as f:
    q_table = pickle.load(f)

for state, q_values in list(q_table.items()):
    print(f"State: {state} → Q-values: {q_values}")
