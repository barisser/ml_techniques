import networkx as nx
import torch
import math
import numpy as np


def make_dag(input_shape, output_shape, capacity, unitsize):
	gr = nx.DiGraph()

	input_units = math.ceil(input_shape.numel() / unitsize)
	output_units = math.ceil(output_shape.numel() / unitsize)

	interior_units = math.ceil(capacity / unitsize)

	total_units = input_units + output_units + interior_units

	adj = np.random.randn(total_units, total_units)

	gr = nx.from_numpy_array(adj)

	return gr





inputsample = torch.randn(32, 784)
output_sample = torch.randn(32, 10)
dag = make_dag(inputsample.shape, output_sample.shape, 4e4, 1000)
import pdb;pdb.set_trace()