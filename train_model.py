import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import prettytensor as pt


#mini batch generator

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):

    m = X.shape[1]
    mini_batches = []
  

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

   
    num_complete_minibatches = math.floor( m / mini_batch_size) 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

   
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

#one hot encoding

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

#flattening of images.important before feeding in to a neural net

X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

#normalise them

X_train = X_train_flatten/255.
X_test = X_test_flatten/255.

#one hot encoding of y

Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)


def sigmoid(z):

    x = tf.placeholder(tf.float32, name="x")

    sigmoid = tf.sigmoid(x)
    sess = tf.Session()
    
    result = sess.run(sigmoid, feed_dict={x: z})
    return result


def cost(logits, labels):
    z = tf.placeholder(tf.float32, name="z")
    y = tf.placeholder(tf.float32, name="y")

  
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

   
    sess = tf.Session()

    cost = sess.run(cost, feed_dict={z: logits, y: labels})

    sess.close()
    return cost


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=(n_x, None), name="X")
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name="Y")
    return X, Y


def initialize_parameters():
    W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())


    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


def forward_propagation(X, parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3
   
    return Z3


def compute_cost(Z3, Y):

 
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

  
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
 

    return cost


#model trained on 2500 iterations on batch size of 32

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,num_epochs=2500, minibatch_size=32, print_cost=True):

    ops.reset_default_graph() 
 
    (n_x, m) = X_train.shape  
    n_y = Y_train.shape[0] 
    costs = [] 
    
    X, Y = create_placeholders(n_x, n_y)
   
    parameters = initialize_parameters()
  
    Z3 = forward_propagation(X, parameters)
   
    cost = compute_cost(Z3, Y)
   
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
 
    init = tf.global_variables_initializer()

 
    with tf.Session() as sess:

       
        sess.run(init)
        
        
        for epoch in range(num_epochs):

            epoch_cost = 0.  
            num_minibatches = int(m / minibatch_size) 
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches

         
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

    
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

      
        parameters = sess.run(parameters)
        
        #save our model
        
        saver = tf.train.Saver()
        saver.save(sess, 'D:\salman\kaifmodel')
        
        print("Parameters have been trained!")

        
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters

parameters = model(X_train, Y_train, X_test, Y_test)
