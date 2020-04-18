import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense


# t, x === [0,1]X[0,1]
# du/dt = (nu * d2u/dx2) - (alpha * u * du/dx)
# u(t, x==0) = a
# u(t ,x==1) = b
# u(t==0, x) = u0(x) Linearfunction

def create_dataset():
    # t: [0, 1] * (f, g, u)
    # x: [0, 1] * (f, g, u)
    # nu: [1e-2, 1e-1] Viscosity
    # alp: [1e-2, 1]
    # a: [-1, 1] at x==0
    # b: [-1, 1] at x==1
    # gx: g(x) value of boundary condition
    # u0: u0(x) value of initial condition - linear function
    # u0(x) = y2-y1/x2-x1*(x-x1) + y1 -- y1=a, y2=b, x1=0, x2=1, x=u at t=0

    f = tf.random.uniform(shape=[20000, 2], minval=0., maxval=1.)
    g_1 = tf.random.uniform(shape=[20000], minval=0., maxval=1.)
    g_2 = tf.concat([tf.zeros([10000]), tf.ones([10000])], axis=0)
    g = tf.stack([g_1, g_2], axis=1)
    u = tf.random.uniform(shape=[20000, 2], minval=0., maxval=1.)*[0., 1.]
    nu = tf.random.uniform(shape=[20000], minval=1e-2, maxval=1e-1)
    alp = tf.random.uniform(shape=[20000], minval=1e-2, maxval=1.)
    a = tf.random.uniform(shape=[20000], minval=-1., maxval=1.)
    b = tf.random.uniform(shape=[20000], minval=-1., maxval=1.)
    gx = tf.concat([a[:10000], b[10000:]], axis=0)
    u0 = (b-a)*u[:, 1] + a

    na = tf.stack([nu, alp], axis=1)
    ab = tf.stack([a,b], axis=1)
    f = tf.concat([f, na, ab], axis=1)
    g = tf.concat([g, na, ab], axis=1)
    u = tf.concat([u, na, ab], axis=1)

    batch_size = 10
    train_dataset = tf.data.Dataset.from_tensor_slices((f, g, u, nu, alp, gx, u0))
    train_dataset = train_dataset.shuffle(buffer_size=20000).batch(batch_size)

    return train_dataset


def create_model():
    model = Sequential()
    model.add(Dense(6, name='digits'))
    layers = [Dense(10, activation='tanh') for _ in range(6)]
    outputs = [Dense(1, activation='tanh', name='predictions')]
    layers = layers + outputs
    perceptron = Sequential(layers)
    model.add(perceptron)
    return model


def run(model, batch):
    # batch = [f, g, u]
    # f.shape = (batch_size, 6) {t, x, nu, alp, a, b}

    with tf.GradientTape() as t:
        t.watch(batch[0])
        with tf.GradientTape() as tape:
            tape.watch(batch[0])
            f_logits = model(batch[0])
        df = tape.gradient(f_logits, batch[0])[:, :2]
        dfdt = df[:, 0]
        dfdx = df[:, 1]
    dfdx2 = t.gradient(dfdx, batch[0])[:, 1]

    g_logits = model(batch[1])
    u_logits = model(batch[2])

    f_logits = tf.squeeze(f_logits)
    g_logits = tf.squeeze(g_logits)
    u_logits = tf.squeeze(u_logits)

    return (dfdt, dfdx, dfdx2), (f_logits, g_logits, u_logits)


def loss_fn(diff, logits, vals):
    # J(theta) = ||df/dt - Lf||^2 + ||f - g(x)||^2 + ||f - u0(x)||^2
    # Lf = nu * d2f/dx2 - alpha * f * df/dx

    # vals = [nu, alpha, g(x), u0(x)]

    l1 = diff[0] + vals[1] * logits[0] * diff[1] - vals[0] * diff[2]
    l2 = logits[1] - vals[2]
    l3 = logits[2] - vals[3]
    l1 = tf.math.reduce_sum(tf.math.square(l1))
    l2 = tf.math.reduce_sum(tf.math.square(l2))
    l3 = tf.math.reduce_sum(tf.math.square(l3))
    return l1 + l2 + l3


def train():
    model = create_model()
    print("Model Created")
    train_dataset = create_dataset()
    epochs = 100
    lowest_loss = 5.0
    optimizer = tf.keras.optimizers.Adam(0.0001)
    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch))
        print('Lowest Loss till now %s' %(lowest_loss))
        for step, batch in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                diff, logits = run(model, batch[:3])
                loss_value = loss_fn(diff, logits, batch[3:])
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if step % 75 == 0:
                if lowest_loss >= float(loss_value):
                    lowest_loss = float(loss_value)
                print('Training loss (for one batch) at step %s: %s' %
                    (step, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * 10))
    model.save('my_model.h5')
    return model

def plot(model):
    x = tf.linspace(0., 1., 60)
    t = tf.ones_like(x)
    nu = tf.ones_like(x) * 0.02
    alp = tf.ones_like(x) * 0.95
    a = tf.ones_like(x) * 0.9
    b = tf.ones_like(x) * -0.9
    X = tf.stack([t,x,nu,alp,a,b], axis=1)
    F = model(X)

    plt.figure()
    plt.plot(x, F)
    plt.savefig('fig')


if __name__ == "__main__":
    model = train()
    plot(model)
