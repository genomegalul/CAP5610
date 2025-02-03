import numpy as np
import pandas as pd
from graphviz import Digraph

# Given dataset.
data = pd.DataFrame({
    "Early": [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1],
    "Finished_HMK": [1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0],
    "Senior": [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
    "Likes_Coffee": [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1],
    "Liked_The_Last_Jedi": [1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0],
    "A": [1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1]
})

# Function to compute entropy.
def entropy(subset):
    p_pos = subset["A"].mean()
    p_neg = 1 - p_pos
    return - (p_pos * np.log2(p_pos) if p_pos > 0 else 0) - (p_neg * np.log2(p_neg) if p_neg > 0 else 0)

# Computing dataset entropy.
entropy_S = entropy(data)

# Computing information gain for each attribute.
def compute_info_gain(data, attribute, entropy_parent):
    values = data[attribute].unique()
    weighted_entropy = sum(
        (len(data[data[attribute] == v]) / len(data)) * entropy(data[data[attribute] == v]) for v in values
    )
    return entropy_parent - weighted_entropy

# Compute information gain for first depth.
info_gains = {col: compute_info_gain(data, col, entropy_S) for col in data.columns[:-1]}
best_split = max(info_gains, key=info_gains.get)

# First split.
subset_yes = data[data[best_split] == 1]
subset_no = data[data[best_split] == 0]
entropy_yes = entropy(subset_yes)
entropy_no = entropy(subset_no)

# Computing information gain for second depth.
info_gains_yes = {col: compute_info_gain(subset_yes, col, entropy_yes) for col in subset_yes.columns[:-1]}
info_gains_no = {col: compute_info_gain(subset_no, col, entropy_no) for col in subset_no.columns[:-1]}
best_split_yes = max(info_gains_yes, key=info_gains_yes.get)
best_split_no = max(info_gains_no, key=info_gains_no.get)

# Second splits.
subset_yes_yes = subset_yes[subset_yes[best_split_yes] == 1]
subset_yes_no = subset_yes[subset_yes[best_split_yes] == 0]
subset_no_yes = subset_no[subset_no[best_split_no] == 1]
subset_no_no = subset_no[subset_no[best_split_no] == 0]

entropy_yes_yes = entropy(subset_yes_yes)
entropy_yes_no = entropy(subset_yes_no)
entropy_no_yes = entropy(subset_no_yes)
entropy_no_no = entropy(subset_no_no)

# Computing information gain for third depth.
info_gains_yes_yes = {col: compute_info_gain(subset_yes_yes, col, entropy_yes_yes) for col in subset_yes_yes.columns[:-1]}
info_gains_yes_no = {col: compute_info_gain(subset_yes_no, col, entropy_yes_no) for col in subset_yes_no.columns[:-1]}
info_gains_no_yes = {col: compute_info_gain(subset_no_yes, col, entropy_no_yes) for col in subset_no_yes.columns[:-1]}
info_gains_no_no = {col: compute_info_gain(subset_no_no, col, entropy_no_no) for col in subset_no_no.columns[:-1]}

best_split_yes_yes = max(info_gains_yes_yes, key=info_gains_yes_yes.get)
best_split_yes_no = max(info_gains_yes_no, key=info_gains_yes_no.get)
best_split_no_yes = max(info_gains_no_yes, key=info_gains_no_yes.get)
best_split_no_no = max(info_gains_no_no, key=info_gains_no_no.get)

# Function to create visualizations of the trees as images.
def create_decision_tree(depth):
    dot = Digraph()

    # Depth 1
    if depth == 1:
        # Root node
        dot.node("A", f'{best_split}?\nEntropy: {entropy_S:.3f}\n(8+, 7-)')
        
        # Leaf nodes
        dot.node("B", f'Leaf\n({len(subset_yes[subset_yes["A"] == 1])}+, {len(subset_yes[subset_yes["A"] == 0])}-)\nEntropy: {entropy_yes:.3f}', shape='box')
        dot.node("C", f'Leaf\n({len(subset_no[subset_no["A"] == 1])}+, {len(subset_no[subset_no["A"] == 0])}-)\nEntropy: {entropy_no:.3f}', shape='box')
        dot.edge("A", "B", label="Yes")
        dot.edge("A", "C", label="No")
        
        dot.render('DT1', format='png', cleanup=True)

    # Depth 2
    elif depth == 2:
        # Root node
        dot.node("A", f'Finished_HMK?\nEntropy: {entropy_S:.3f}\n(8+, 7-)')

        # Level 1 split
        dot.node("B", f'Liked_The_Last_Jedi?\nEntropy: {entropy_yes:.3f}\n({len(subset_yes[subset_yes["A"] == 1])}+, {len(subset_yes[subset_yes["A"] == 0])}-)')
        dot.node("C", f'Likes_Coffee?\nEntropy: {entropy_no:.3f}\n({len(subset_no[subset_no["A"] == 1])}+, {len(subset_no[subset_no["A"] == 0])}-)')
        dot.edge("A", "B", label="Yes")
        dot.edge("A", "C", label="No")

        # Leaf nodes
        dot.node("D", f'Leaf\n({len(subset_yes_yes[subset_yes_yes["A"] == 1])}+, {len(subset_yes_yes[subset_yes_yes["A"] == 0])}-)\nEntropy: {entropy_yes_yes:.3f}', shape='box')
        dot.node("E", f'Leaf\n({len(subset_yes_no[subset_yes_no["A"] == 1])}+, {len(subset_yes_no[subset_yes_no["A"] == 0])}-)\nEntropy: {entropy_yes_no:.3f}', shape='box')
        dot.node("F", f'Leaf\n({len(subset_no_yes[subset_no_yes["A"] == 1])}+, {len(subset_no_yes[subset_no_yes["A"] == 0])}-)\nEntropy: {entropy_no_yes:.3f}', shape='box')
        dot.node("G", f'Leaf\n({len(subset_no_no[subset_no_no["A"] == 1])}+, {len(subset_no_no[subset_no_no["A"] == 0])}-)\nEntropy: {entropy_no_no:.3f}', shape='box')
        dot.edge("B", "D", label="Yes")
        dot.edge("B", "E", label="No")
        dot.edge("C", "F", label="Yes")
        dot.edge("C", "G", label="No")

        dot.render('DT2', format='png', cleanup=True)

    # Depth 3
    elif depth == 3:
        # Root node
        dot.node("A", f'Finished_HMK?\nEntropy: {entropy_S:.3f}\n(8+, 7-)')

        # Level 1 split
        dot.node("B", f'Liked_The_Last_Jedi?\nEntropy: {entropy_yes:.3f}\n({len(subset_yes[subset_yes["A"] == 1])}+, {len(subset_yes[subset_yes["A"] == 0])}-)')
        dot.node("C", f'Likes_Coffee?\nEntropy: {entropy_no:.3f}\n({len(subset_no[subset_no["A"] == 1])}+, {len(subset_no[subset_no["A"] == 0])}-)')
        dot.edge("A", "B", label="Yes")
        dot.edge("A", "C", label="No")

        # Level 2 split
        dot.node("D", f'{best_split_yes_yes}?\nEntropy: {entropy_yes_yes:.3f}\n({len(subset_yes_yes[subset_yes_yes["A"] == 1])}+, {len(subset_yes_yes[subset_yes_yes["A"] == 0])}-)')
        dot.node("E", f'{best_split_yes_no}?\nEntropy: {entropy_yes_no:.3f}\n({len(subset_yes_no[subset_yes_no["A"] == 1])}+, {len(subset_yes_no[subset_yes_no["A"] == 0])}-)')
        dot.node("F", f'{best_split_no_yes}?\nEntropy: {entropy_no_yes:.3f}\n({len(subset_no_yes[subset_no_yes["A"] == 1])}+, {len(subset_no_yes[subset_no_yes["A"] == 0])}-)')
        dot.node("G", f'{best_split_no_no}?\nEntropy: {entropy_no_no:.3f}\n({len(subset_no_no[subset_no_no["A"] == 1])}+, {len(subset_no_no[subset_no_no["A"] == 0])}-)')
        dot.edge("B", "D", label="Yes")
        dot.edge("B", "E", label="No")
        dot.edge("C", "F", label="Yes")
        dot.edge("C", "G", label="No")

        # Leaf nodes
        dot.node("H", f'Leaf\n({len(subset_yes_yes[subset_yes_yes[best_split_yes_yes] == 1]["A"] == 1)}+, {len(subset_yes_yes[subset_yes_yes[best_split_yes_yes] == 0]["A"] == 0)}-)\nEntropy: {entropy(subset_yes_yes):.3f}', shape='box')
        dot.node("I", f'Leaf\n({len(subset_yes_yes[subset_yes_yes[best_split_yes_yes] == 0]["A"] == 1)}+, {len(subset_yes_yes[subset_yes_yes[best_split_yes_yes] == 0]["A"] == 0)}-)\nEntropy: {entropy(subset_yes_yes):.3f}', shape='box')
        dot.node("J", f'Leaf\n({len(subset_yes_no[subset_yes_no[best_split_yes_no] == 1]["A"] == 1)}+, {len(subset_yes_no[subset_yes_no[best_split_yes_no] == 0]["A"] == 0)}-)\nEntropy: {entropy(subset_yes_no):.3f}', shape='box')
        dot.node("K", f'Leaf\n({len(subset_yes_no[subset_yes_no[best_split_yes_no] == 0]["A"] == 1)}+, {len(subset_yes_no[subset_yes_no[best_split_yes_no] == 0]["A"] == 0)}-)\nEntropy: {entropy(subset_yes_no):.3f}', shape='box')
        dot.node("L", f'Leaf\n({len(subset_no_yes[subset_no_yes[best_split_no_yes] == 1]["A"] == 1)}+, {len(subset_no_yes[subset_no_yes[best_split_no_yes] == 0]["A"] == 0)}-)\nEntropy: {entropy(subset_no_yes):.3f}', shape='box')
        dot.node("M", f'Leaf\n({len(subset_no_yes[subset_no_yes[best_split_no_yes] == 0]["A"] == 1)}+, {len(subset_no_yes[subset_no_yes[best_split_no_yes] == 0]["A"] == 0)}-)\nEntropy: {entropy(subset_no_yes):.3f}', shape='box')
        dot.node("N", f'Leaf\n({len(subset_no_no[subset_no_no[best_split_no_no] == 1]["A"] == 1)}+, {len(subset_no_no[subset_no_no[best_split_no_no] == 0]["A"] == 0)}-)\nEntropy: {entropy(subset_no_no):.3f}', shape='box')
        dot.node("O", f'Leaf\n({len(subset_no_no[subset_no_no[best_split_no_no] == 0]["A"] == 1)}+, {len(subset_no_no[subset_no_no[best_split_no_no] == 0]["A"] == 0)}-)\nEntropy: {entropy(subset_no_no):.3f}', shape='box')
        dot.edge("D", "H", label="Yes")
        dot.edge("D", "I", label="No")
        dot.edge("E", "J", label="Yes")
        dot.edge("E", "K", label="No")
        dot.edge("F", "L", label="Yes")
        dot.edge("F", "M", label="No")
        dot.edge("G", "N", label="Yes")
        dot.edge("G", "O", label="No")

        dot.render('DT3', format='png', cleanup=True)

# Generating the decision trees.
create_decision_tree(1)
create_decision_tree(2)
create_decision_tree(3)
