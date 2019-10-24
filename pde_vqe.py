import pennylane as qml
import tensorflow as tf

num_wires = 2

dev = qml.device("default.qubit", wires=num_wires)

@qml.qnode(dev)
def circuit():
    pass

"""
The basic premise of this variational quantum circuit is to solve Partial Differential Equations.
The process can be described as below:
1. A quantum circuit is prepared which has the number of inputs and outputs equal to the PDE being
   modelled((t,x) and u for 1D Burgers' Equation). (Don't know which gates to use)
2. This quantum circuit is to be run to generate the NN, NN_shift and NN_init output, based on 3
   different sets of inputs, and cost is to be calculated based on the DGM formula 13
2. This quantum circuit is then "trained" using its trainable parameters using AdamOptimizer
3. Finally the desired output should be received from this VQE
"""