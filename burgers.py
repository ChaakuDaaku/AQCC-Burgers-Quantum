import matplotlib.pyplot as plt
import numpy as np
import sympy
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense


def create_dataset():
    f = tf.random.uniform((20000, 2)) * ([2 * np.pi, 0.5])
    g = tf.random.uniform((20000, 2)) * ([0., 0.5])
    u = tf.random.uniform((20000, 2)) * ([2 * np.pi, 0.])

    with tf.GradientTape() as tape:
        tape.watch(u)
        phi1 = tf.math.exp(-tf.math.square(u[:,0]-4*u[:,1])/(0.2*(u[:,1]+1)))
        phi2 = tf.math.exp(-tf.math.square(u[:,0]-4*u[:,1]-2*np.pi)/(0.2*(u[:,1]+1)))
        phi = phi1 + phi2
    dphi = tape.gradient(phi, u)

    dphidx = dphi[:,0]
    u0 = -(2*0.05/phi*dphidx)+4.0

    batch_size = 64
    train_dataset = tf.data.Dataset.from_tensor_slices((f, g, u, u0))
    train_dataset = train_dataset.shuffle(buffer_size=20000).batch(batch_size)

    return train_dataset


def create_model():
    inputs = Input(shape=(2,), name='digits')
    x = Dense(10, activation='tanh', name='dense_1')(inputs)
    x = Dense(10, activation='tanh', name='dense_2')(x)
    x = Dense(10, activation='tanh', name='dense_3')(x)
    x = Dense(10, activation='tanh', name='dense_4')(x)
    x = Dense(10, activation='tanh', name='dense_5')(x)
    outputs = Dense(1, activation='tanh', name='predictions')(x)
    model = Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(0.01)

    return model, optimizer


def run(model, f_batch, g_batch, u_batch):
    with tf.GradientTape() as t:
        t.watch(f_batch)
        with tf.GradientTape() as tape:
            tape.watch(f_batch)
            f_logits = model(f_batch)
        df = tape.gradient(f_logits, f_batch)
        dfdx = df[:, 0]
        dfdt = df[:, 1]
    dfdx2 = t.gradient(dfdx, f_batch)[:, 0]

    g_logits = model(g_batch)
    u_logits = model(u_batch)

    f_logits = tf.squeeze(f_logits)
    g_logits = tf.squeeze(g_logits)
    u_logits = tf.squeeze(u_logits)

    return (dfdt, dfdx, dfdx2), (f_logits, g_logits, u_logits)


def loss_fn(u0_batch, diff, logits):
    l1 = diff[0] + tf.math.multiply(logits[0], diff[1]) - 0.05 * diff[2]
    l2 = logits[1] - logits[0]
    l3 = logits[2] - u0_batch
    l1 = tf.math.reduce_mean(tf.math.square(l1))
    l2 = tf.math.reduce_mean(tf.math.square(l2))
    l3 = tf.math.reduce_mean(tf.math.square(l3))
    return 0.49*l1 + 0.01*l2 + 0.5*l3

def train():
    model, optimizer = create_model()
    print("Model Created")

    epochs = 10
    for epoch in range(epochs):
        train_dataset = create_dataset()
        print('Start of epoch %d' % (epoch,))
        for step, (f_batch, g_batch, u_batch, u0_batch) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                diff, logits = run(model, f_batch, g_batch, u_batch)
                loss_value = loss_fn(u0_batch, diff, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if step % 20 == 0:
                print('Training loss (for one batch) at step %s: %s' %
                    (step, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * 64))

    return model

def analytic_method():

    X, T, NU = sympy.symbols('X T NU')
    phi = sympy.exp(-(X - 4 * T) ** 2 / (4 * NU * (T + 1))) + \
        sympy.exp(-(X - 4 * T - 2 * np.pi) ** 2 / (4 * NU * (T + 1)))
    dphidx = phi.diff(X)

    u_analytic_ = -2 * NU / phi * dphidx + 4
    u_analytic = sympy.utilities.lambdify((X, T, NU), u_analytic_)

    return u_analytic


def plot(model):
    instants = [0.0, 0.25, 0.5]
    xtest = np.linspace(0, 2 * np.pi, num=60, dtype=np.float32)
    analytic = analytic_method()
    plt.figure()
    analytic_plot, ML_plot = [], []
    colors = ['r', 'b', 'g']

    for instant in instants:

        ttest = np.ones(xtest.shape, dtype=np.float32) * instant
        xt = tf.convert_to_tensor(np.column_stack((xtest, ttest)))
        u_NN = model(xt)
        u_analytic = analytic(xtest, ttest, 0.05)

        current_ML_plot, = plt.plot(
            xtest, u_NN, '.-', color=colors[instants.index(instant)])
        current_analytic_plot, = plt.plot(
            xtest, u_analytic, '-', color=colors[instants.index(instant)])

        analytic_plot.append(current_analytic_plot)
        ML_plot.append(current_ML_plot)

    all_plots = [analytic_plot, ML_plot]
    legend_time = plt.legend(
        all_plots[0], ['t=0.0', 't=0.25', 't=0.5'], loc='upper left', numpoints=1)
    plt.legend([method[0] for method in all_plots], ['Analytic', 'ML'])
    plt.gca().add_artist(legend_time)
    plt.grid(True)
    plt.xlabel(r'$X[-]$')
    plt.ylabel(r'$U[-]$')
    plt.legend(fontsize=8, loc='lower left', numpoints=1)
    plt.savefig('NN_visc_6x10_epoch_2000.pdf', format='pdf')


if __name__ == "__main__":
    model = train()
    plot(model)
