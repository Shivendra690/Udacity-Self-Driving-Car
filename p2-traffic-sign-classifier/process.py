import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from scipy import misc
import sys

print("Initializing....")
def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma), name="conv1_W")
    conv1_b = tf.Variable(tf.zeros(6), name="conv1_b")
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID', name="conv1_before_activation") + conv1_b
    
    # Activation.
    conv1 = tf.nn.relu(conv1, name="conv1_after_activation")

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name="conv1_after_pooling")
    

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma), name="conv2_W")
    conv2_b = tf.Variable(tf.zeros(16), name="conv2_b")
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID', name="conv2_before_activation") + conv2_b
    
    # Activation.
    conv2 = tf.nn.relu(conv2, name="conv2_after_activation")

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name="conv2_before_pooling")

    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma), name="fc1_W")
    fc1_b = tf.Variable(tf.zeros(120), name="fc1_b")
    fc1   = tf.matmul(fc0, fc1_W, name="fc1_before_activation") + fc1_b
    
    # Activation.
    fc1    = tf.nn.relu(fc1, name="fc1_after_activation")

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma), name="fc2_W")
    fc2_b  = tf.Variable(tf.zeros(84), name="fc2_b")
    fc2    = tf.matmul(fc1, fc2_W, name="fc2_before_activation") + fc2_b
    
    # Activation.
    fc2    = tf.nn.relu(fc2, name="fc2_after_activation")
    
    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma), name="fc3_W")
    fc3_b  = tf.Variable(tf.zeros(43), name="fc3_b")
    logits = tf.matmul(fc2, fc3_W, name="logits") + fc3_b
    
    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 1), name="x")
y = tf.placeholder(tf.int32, (None), name="y")
one_hot_y = tf.one_hot(y, 43, name="one_hot_y")

rate = 0.02

# Declare Variables
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits, name="cross_entropy")
loss_operation = tf.reduce_mean(cross_entropy, name="loss_operation")
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation, name="training")
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1), name="correct_prediction")
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy_operation")
graph = tf.Graph()
saver = tf.train.Saver()

def process_data(data):
    gray_data = np.zeros(shape=(data.shape[0], data.shape[1], data.shape[2], 1))
    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                gray_data[i][j][k] = np.array([(np.mean(data[i][j][k]) - 128) / 128])
    return gray_data

print("Importing labels for classification")
int_to_labels = {}
with open('signnames.csv', 'r') as file:
    line = file.readline().strip()
    line = file.readline().strip()
    while line != '':
        n, label = line.split(",")
        int_to_labels[int(n)] = label
        line = file.readline().strip()

if __name__ == '__main__':
    pictures = []
    imgs = []
    print("All images must be preprocessed as 32px by 32px!!")
    while True:
        path = input("\nInput path to image (with extension): ").strip()
        if path.strip() != ':done' and path.strip() != ':exit':
            pictures.append(path)
            try:
                img = misc.imread(path)
            except Exception as e:
                print("No such image. Check ur path")
                continue
            w, h, c = img.shape
            if w >= 32 or h >= 32:
                print("this image has wrong dimensions: {0}x{1}x{2}".format(w,g,c))
                print("All images must be preprocessed as 32px by 32px!!")
            else:
                img = np.pad(img, (((32-w)//2,(32-w)//2),((32-h)//2,(32-h)//2),(0,0)), 'constant')
                imgs.append(img)
        else:
            break
    pictures = np.array(pictures)
    imgs = np.array(imgs)
    procesed_imgs = process_data(imgs)
    print("Start classifying. It may take a while")
    with tf.Session() as sess:
        saver.restore(sess, "./lenet")
        lgts = sess.run(logits, feed_dict={x:procesed_imgs})
        preds = sess.run(tf.argmax(lgts, axis=1))
        for i, pic in enumerate(imgs):
            print("Classifying image:", pictures[i])
            print("Prediction:" + int_to_labels[preds[i]])
