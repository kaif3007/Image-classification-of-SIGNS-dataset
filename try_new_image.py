import scipy
from PIL import Image
import tensorflow as tf
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt


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

def predict(X, parameters):
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}

    x = tf.placeholder("float", [12288, 1])

    z3 = forward_propagation(x, params)
    p = tf.argmax(z3)

    with tf.Session() as sess:
        prediction = sess.run(p, feed_dict={x: X})

    return prediction


#restoring our model along with its weights

params={}
with tf.Session() as sess:

    saver = tf.train.import_meta_graph('kaifmodel.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    
    W1=sess.run('W1:0')
    W2=sess.run('W2:0')
    W3 = sess.run('W3:0')
    b1 = sess.run('b1:0')
    b2 = sess.run('b2:0')
    b3 = sess.run('b3:0')
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}


#trying new image

my_image = "xc.jpg"


fname = my_image
image = np.array(ndimage.imread(fname, flatten=False))

my_image = scipy.misc.imresize(image, size=(64, 64)).reshape((1, 64 * 64 * 3)).T

my_image_prediction = predict(my_image, params)


plt.imshow(image)
plt.show()
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
