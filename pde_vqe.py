import pennylane as qml
import tensorflow as tf

num_wires = 2

dev = qml.device("default.qubit", wires=num_wires)

@qml.qnode(dev)
def circuit():
    pass
