"""
Dynamic Arch


We want to discover the network topology.  We represent the neural network as a DAG of nodes.  Edges between nodes are 'compute units' aka dense layers.

We will start with a semi random DAG topology.  This will form a messy neural network.  The values of the adjacency matrix composing the DAG will also be trainable parameters.  These values will modulate the strength of information passed along by the corresponding neural network weights.  

We will introduce a regularization term on this adjacency matrix to try to prune edges in the DAG.  The regularization term should drive strength to zero, for example -1/|X| norm pruning.

We will execute the graph in semi topological order.  We start with input nodes.  We basically perform BFS starting from input nodes.  Nodes that have unreachable dependencies can still be triggered.
"""