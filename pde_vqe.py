import pennylane as qml
import tensorflow as tf

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

import pennylane as qml
import tensorflow as tf
import time
import numpy as np
# Numerics
nu = 0.05  # Viscosity of Burgers' equation
T = 0.5   # Final time of the simulation

# Training settings
nSamples = 20000         # Number of random points to be sampled
epochs = 2000            # Number of epochs
learning_rate = 1.0e-3   # Learning rate
num_layers = 4

# Visualization parameters
instants = [0.0, 0.25, 0.5]  # Time instants for plotting

dev = qml.device("strawberryfields.fock", wires=2, cutoff_dim=10)


def layer(v):
    qml.Beamsplitter(v[0], v[1], wires=[0, 1])
    qml.Rotation(v[2], wires=0)
    qml.Rotation(v[3], wires=1)
    qml.Squeezing(v[4], 0.0, wires=0)
    qml.Squeezing(v[5], 0.0, wires=1)
    qml.Beamsplitter(v[6], v[7], wires=[0, 1])
    qml.Rotation(v[8], wires=0)
    qml.Rotation(v[9], wires=1)
    qml.Displacement(v[10], 0.0, wires=0)
    qml.Displacement(v[11], 0.0, wires=1)
    qml.Kerr(v[12], wires=0)
    qml.Kerr(v[13], wires=1)


@qml.qnode(dev, interface="tf")
def circuit(var, x=None, num_wires=2):
    # Encode input x into quantum state
    qml.Displacement(x[:, 0], 0.0, wires=0)
    qml.Displacement(x[:, 1], 0.0, wires=1)

    # "layer" subcircuits
    for v in var:
        layer(v)

    return qml.expval(qml.X(0))


def train(costs):
    return 0.49*tf.reduce_sum(input_tensor=tf.square(costs[0])) +\
        0.01*tf.reduce_sum(input_tensor=tf.square(costs[1])) +\
        0.5*tf.reduce_sum(input_tensor=tf.square(costs[2]))


def cost(var, feats):

    phi = tf.exp(-(feats[2][:, 0]-4*feats[2][:, 1])**2/(4*nu*(feats[2][:, 1]+1))) +\
        tf.exp(-(feats[2][:, 0]-4*feats[2][:, 1]-2*np.pi)
               ** 2/(4*nu*(feats[2][:, 1]+1)))
    dphidx = tf.gradients(ys=phi, xs=feats[2])[0][:, 0]
    U0 = -(2*nu/phi*dphidx)+4.0
    U0 = tf.squeeze(U0)
    preds = [circuit(var, f) for f in feats]
    dQNdx = tf.gradients(ys=preds[0], xs=feats[0])[0][:, 0]
    dQNdt = tf.gradients(ys=preds[0], xs=feats[0])[0][:, 1]
    d2QNdx2 = tf.gradients(ys=dQNdx, xs=feats[0])[0][:, 0]

    costs = [
        dQNdt+tf.multiply(preds[0], dQNdx)-nu*d2QNdx2,
        preds[1]-preds[0],
        preds[2]-U0
    ]
    return loss(costs)


def create_mini_batch(X, batch_size=256):
    X_MB = []
    mb_index = np.random.choice(len(X), len(X), replace=False)
    X_shuffled = X[mb_index, :]
    for i in range(0, len(X), batch_size):
        X_MB.append(tf.Variable(X_shuffled[i:i+batch_size, :]))
    return X_MB

if __name__ == "__main__":
    var = tf.Variable(0.05*tf.random.normal(num_layers, 14))
    opt = tf.keras.optimizers.Adam(learning_rate)
    for epoch in np.arange(epochs):
        # Get random points inside domain
        inside_input = np.random.rand(nSamples, 2) * ([2 * np.pi, T])
        boundary_input = np.random.rand(nSamples, 2) * ([0.0, T])
        initial_input = np.random.rand(nSamples, 2) * ([2 * np.pi, 0.0])

        inside_mb = create_mini_batch(inside_input)
        boundary_mb = create_mini_batch(boundary_input)
        initial_mb = create_mini_batch(initial_input)

        for i_mb in range(len(inside_mb)):
            epoch_data = [inside_mb[i_mb], boundary_mb[i_mb], initial_mb[i_mb]]
            var = opt.step(lambda v: cost(v, epoch_data), var)
