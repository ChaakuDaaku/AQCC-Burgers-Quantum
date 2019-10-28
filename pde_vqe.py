import pennylane as qml
import tensorflow as tf
import numpy as np
import time
import sympy
import matplotlib.pyplot as plt

nu = 0.05
T = 0.5
num_layers = 4
nSamples = 2000
epochs = 200
learning_rate = 1.0e-4
instants = [0.0, 0.25, 0.5]

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
def circuit(var, x):
    qml.Displacement(x[0], 0.0, wires=0)
    qml.Displacement(x[1], 0.0, wires=1)

    for i in range(var.shape[0]):
        layer(var[i])

    return qml.expval(qml.X(0))


def create_mini_batch(X, batch_size=256):
    X_MB = []
    mb_index = np.random.choice(len(X), len(X), replace=False)
    X_shuffled = X[mb_index, :]

    for i in range(0, len(X), batch_size):
        X_MB.append(X_shuffled[i: i+batch_size, :])

    return X_MB


def j_theta(var, f, g, h):
    phi = tf.math.exp(-(h[0]-4*h[1])**2/(4*nu*(h[1]+1)))
    qn_shift = circuit(var, g)
    qn_init = circuit(var, h)
    with tf.GradientTape() as tape:
        phi += tf.math.exp(-(h[0]-4*h[1]-2*np.pi)**2/(4*nu*(h[1]+1)))
    dphidx, dphidt = tape.gradient(phi, h)
    u0 = -(2 * nu / phi * dphidx) + 4.0
    with tf.GradientTape() as tape:
        with tf.GradientTape() as tt:
            qn = circuit(var, f)
        _, input_grad = tt.gradient(qn, [var, f])
        dqndx, dqndt = input_grad[0], input_grad[1]
    _, input_grad2 = tape.gradient(dqndx, [var, f])
    d2qndx2 = input_grad2[0]
    c1 = (dqndt + (qn*dqndx - nu*d2qndx2))**2
    c2 = (qn_shift - qn)**2
    c3 = (qn_init - u0)**2
    return c1, c2, c3


def analytic_method():

    X, T, NU = sympy.symbols('X T NU')
    phi = sympy.exp(-(X - 4 * T) ** 2 / (4 * NU * (T + 1))) + \
        sympy.exp(-(X - 4 * T - 2 * np.pi) ** 2 / (4 * NU * (T + 1)))
    dphidx = phi.diff(X)

    u_analytic_ = -2 * NU / phi * dphidx + 4
    u_analytic = sympy.utilities.lambdify((X, T, NU), u_analytic_)

    return u_analytic


if __name__ == "__main__":
    var = tf.Variable(tf.random.normal((num_layers, 14), mean=0.0,
                                       stddev=1.0), trainable=True)
    print("Start Training")
    start_time = time.time()

    for epoch in np.arange(epochs):
        inside_input = np.random.rand(nSamples, 2) * ([2 * np.pi, T])
        boundary_input = np.random.rand(nSamples, 2) * ([0.0, T])
        initial_input = np.random.rand(nSamples, 2) * ([2 * np.pi, 0.0])

        inside_mb = create_mini_batch(inside_input)
        boundary_mb = create_mini_batch(boundary_input)
        initial_mb = create_mini_batch(initial_input)

        for i, j, k in zip(inside_mb, boundary_mb, initial_mb):
            # Update weights over 79 mini batches
            cost1, cost2, cost3 = 0.0, 0.0, 0.0
            for f, g, h in zip(i, j, k):
                # Sum over mini batch of 256
                f, g, h = tf.Variable(f), tf.Variable(g), tf.Variable(h)
                c1, c2, c3 = j_theta(var, f, g, h)
                cost1, cost2, cost3 = cost1 + c1, cost2 + c2, cost3 + c3
            loss = 0.49 * cost1 + 0.01 * cost2 + 0.5 * cost3
            opt_op = tf.keras.optimizers.Adam(learning_rate).minimize(loss)
            opt_op.run()

        if epoch % 10 == 0:
            tc1, tc2, tc3 = 0.0, 0.0, 0.0
            # Calculate loss and print
            for f, g, h in zip(inside_input, boundary_input, initial_input):
                f, g, h = tf.Variable(f), tf.Variable(g), tf.Variable(h)
                c1, c2, c3 = j_theta(var, f, g, h)
                tc1, tc2, tc3 = tc1+c1, tc2+c2, tc3+c3
            l = 0.49*tc1 + 0.01*tc2 + 0.5*tc3
            print(f"Epoch: {epoch}, Loss {l}")

        if epoch == epochs / 2:
            learning_rate = learning_rate / 10

    end_time = time.time()
    print(f"Total Time {end_time - start_time}")

    xtest = np.linspace(0, 2*np.pi, num=60, endpoint=True)
    analytic = analytic_method()

    plt.figure()
    analytic_plot, qml_plot = [], []
    colors = ['r', 'b', 'g']

    for instant in instants:
        ttest = np.ones(xtest.shape)*instant
        xt = np.column_stack((xtest, ttest))
        u_an = analytic(xtest, ttest, nu)
        u_qn = [circuit(var, x) for x in xt]
        cur_qn_plot = plt.plot(
            xtest, u_qn, '.-', color=colors[instants.index(instant)])
        cur_an_plot = plt.plot(
            xtest, u_an, '-', color=colors[instants.index(instant)])
        analytic_plot.append(cur_an_plot)
        qml_plot.append(cur_qn_plot)

    plots = [analytic_plot, qml_plot]
    legend_time = plt.legend(
        plots[0], ['t=0.0', 't=0.25', 't=0.5'], loc='upper left', numpoints=1)
    plt.legend([m[0] for m in plots], ['Analytic', 'QML'])
    plt.gca.add_artist(legend_time)
    plt.grid(True)
    plt.xlabel(r'$X[-]$')
    plt.ylabel(r'$U[-]$')
    plt.legend(fontsize=8, loc='lower left', numpoints=1)
    plt.savefig(f'QNN_epoch_{epoch}.pdf', format='pdf')
