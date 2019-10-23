import pennylane as qml
import tensorflow as tf

nu = 0.05 # Viscosity of Burgers' equation
T  = 0.5   # Final time of the simulationc

# TODO: decide the number of qubits needed assigned to ChaakuDaaku
# TODO: fix the NN flow assigned to swagle8987

# Machine learning
neurons = [10, 10, 10, 10, 10, 10] # Neural network layers
nInputs = 2                  # Dimensionality of Burgers' eq.

# Training settings
nSamples = 20000         # Number of random points to be sampled
epochs = 2000            # Number of epochs
learning_rate = 1.0e-4   # Learning rate
      
dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev, interface="tf")

def qfunc(phi, theta):
    qml.RX(phi[0], wires=0)
    qml.RY(phi[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.PhaseShift(theta, wires=0)
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))


# >>> phi = tf.Variable([0.5, 0.1])
# >>> theta = tf.Variable(0.2)
# >>> circuit1(phi, theta)
def create_NN():
    
     weights = {
               'W1' : tf.Variable(tf.random.normal((nInputs,    neurons[0]), mean = 0.0, stddev = 1.0)), 
               'W2' : tf.Variable(tf.random.normal((neurons[0], neurons[1]), mean = 0.0, stddev = 1.0)),
               'W3' : tf.Variable(tf.random.normal((neurons[1], neurons[2]), mean = 0.0, stddev = 1.0)),
               'W4' : tf.Variable(tf.random.normal((neurons[2], neurons[3]), mean = 0.0, stddev = 1.0)),
               'W5' : tf.Variable(tf.random.normal((neurons[3], neurons[4]), mean = 0.0, stddev = 1.0)),
               'W6' : tf.Variable(tf.random.normal((neurons[4], neurons[5]), mean = 0.0, stddev = 1.0)),
               'WO' : tf.Variable(tf.random.normal((neurons[5], 1),          mean = 0.0, stddev = 1.0))
                }
     
     biases = {
              'b1' : tf.Variable(tf.random.normal(([1]), mean = 0.0, stddev = 1.0)),
              'b2' : tf.Variable(tf.random.normal(([1]), mean = 0.0, stddev = 1.0)),
              'b3' : tf.Variable(tf.random.normal(([1]), mean = 0.0, stddev = 1.0)),       
              'b4' : tf.Variable(tf.random.normal(([1]), mean = 0.0, stddev = 1.0)),
              'b5' : tf.Variable(tf.random.normal(([1]), mean = 0.0, stddev = 1.0)),
              'b6' : tf.Variable(tf.random.normal(([1]), mean = 0.0, stddev = 1.0)),
              'bO' : tf.Variable(tf.random.normal(([1]), mean = 0.0, stddev = 1.0))                  
              }
          
     Input  = [tf.placeholder('float', [None, nInputs]) for _ in range(3)]
     
     h1     = tf.nn.tanh(tf.matmul(Input[0], weights['W1']) + biases['b1'])
     h2     = tf.nn.tanh(tf.matmul(h1,    weights['W2']) + biases['b2'])
     h3     = tf.nn.tanh(tf.matmul(h2,    weights['W3']) + biases['b3'])     
     h4     = tf.nn.tanh(tf.matmul(h3,    weights['W4']) + biases['b4'])    
     h5     = tf.nn.tanh(tf.matmul(h4,    weights['W5']) + biases['b5'])   
     h6     = tf.nn.tanh(tf.matmul(h5,    weights['W6']) + biases['b6'])
     NN     = tf.squeeze(tf.matmul(h6,    weights['WO']) + biases['bO'])
     
     Input_shift  = tf.add(Input[1], tf.constant([2 * np.pi, 0.0]))
     h1_shift     = tf.nn.tanh(tf.matmul(Input_shift, weights['W1']) + biases['b1'])
     h2_shift     = tf.nn.tanh(tf.matmul(h1_shift,    weights['W2']) + biases['b2'])
     h3_shift     = tf.nn.tanh(tf.matmul(h2_shift,    weights['W3']) + biases['b3'])
     h4_shift     = tf.nn.tanh(tf.matmul(h3_shift,    weights['W4']) + biases['b4'])   
     h5_shift     = tf.nn.tanh(tf.matmul(h4_shift,    weights['W5']) + biases['b5']) 
     h6_shift     = tf.nn.tanh(tf.matmul(h5_shift,    weights['W6']) + biases['b6']) 
     NN_shift     = tf.squeeze(tf.matmul(h6_shift,    weights['WO']) + biases['bO']) 
     
     h1_init     = tf.nn.tanh(tf.matmul(Input[2],weights['W1']) + biases['b1'])
     h2_init     = tf.nn.tanh(tf.matmul(h1_init,    weights['W2']) + biases['b2'])
     h3_init     = tf.nn.tanh(tf.matmul(h2_init,    weights['W3']) + biases['b3'])     
     h4_init     = tf.nn.tanh(tf.matmul(h3_init,    weights['W4']) + biases['b4'])    
     h5_init     = tf.nn.tanh(tf.matmul(h4_init,    weights['W5']) + biases['b5'])   
     h6_init     = tf.nn.tanh(tf.matmul(h5_init,    weights['W6']) + biases['b6'])
     NN_init     = tf.squeeze(tf.matmul(h6_init,    weights['WO']) + biases['bO'])     
          
     phi    = tf.exp(-(Input[2][:,0] - 4 * Input[2][:,1]) ** 2 / (4 * nu * (Input[2][:,1] + 1))) + \
              tf.exp(-(Input[2][:,0] - 4 * Input[2][:,1] - 2 * np.pi) ** 2 / (4 * nu * (Input[2][:,1] + 1)))             
     dphidx = tf.gradients(phi, Input[2])[0][:,0]   
     U0     = -(2 * nu / phi * dphidx) + 4.0  
     U0     = tf.squeeze(U0)
     
     dNNdx   = tf.gradients(NN, Input[0])[0][:,0]
     dNNdt   = tf.gradients(NN, Input[0])[0][:,1]
     d2NNdx2 = tf.gradients(dNNdx, Input[0])[0][:,0]    
     cost1   = dNNdt + tf.multiply(NN, dNNdx) - nu * d2NNdx2
    
     cost2   = NN_shift - NN
     cost3   = NN_init  - U0
                   
     return Input, NN, cost1, cost2, cost3

def training_method():
    
    Input, NN, cost1, cost2, cost3 = create_NN()            
    loss =  0.49 * tf.reduce_sum(tf.square(cost1)) + 0.01 * tf.reduce_sum(tf.square(cost2)) + 0.5 * tf.reduce_sum(tf.square(cost3))    
    
    return Input, NN, trainStep, loss


if __name__ == "__main__":

    Input, NN, trainStep, loss = training_method()

    trainStep = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    for i in range(epochs):
        with tf.GradientTape() as tape:
            loss = tf.abs(qfunc(Input, NN) - 0.5)**2
            grads = tape.gradient(loss, [Input, NN])
            
        trainStep.apply_gradients(zip(grads, [Input, NN]), global_step=tf.train.get_or_create_global_step())
