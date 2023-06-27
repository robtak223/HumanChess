import tensorflow as tf
from tensorflow import keras
import numpy as np
import chess.pgn

from utils import *
from error import *


class PolicyNetwork(keras.Model):
    def __init__(self, input_shape, num_actions, num_filters, num_residual_blocks):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(num_filters, kernel_size=3, activation='relu', padding="same")
        self.res_blocks = []
        for _ in range(num_residual_blocks):
            self.res_blocks.append(ResidualBlock(num_filters))
        self.policy_conv = tf.keras.layers.Conv2D(73, kernel_size=1, padding="same", activation="relu")
        self.policy_fc = tf.keras.layers.Dense(num_actions, activation="softmax")
        self.trained = False
        self.build(input_shape)

    def call(self, inputs):
        
        x = self.conv1(inputs)
        for block in self.res_blocks:
            x = block(x)
        x = self.policy_conv(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.policy_fc(x)
        return x
    
    def train_model(self, x_train, y_train, weights_save_file, epochs=11, batch_size=32, show_summary=True, lr=1e-3):
        if type(x_train) != list:
            raise InputException("x_train and y_train must be lists")
        if type(y_train) != list:
            raise InputException("x_train and y_train must be of the same type")
        if len(x_train) == 0:
            raise InputException("length of filenames or arrays must be greater than 0")
        if len(y_train) != len(x_train):
            raise InputException("y_train and x_train must contain the same number of samples")
        self.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=lr), loss='categorical_crossentropy', metrics="accuracy")
        if show_summary:
            self.summary()
        if type(x_train[0]) == "string":
            for i in range(len(x_train)):
                x_train = np.load(x_train[i])
                y_train = np.load(y_train[i])
                self.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
                self.save_weights(weights_save_file)
        self.trained = True
    
    def evaluate_model(self, x_test_file, y_test_file):
        x_train2 = np.load(x_test_file)
        y_train2 = np.load(y_test_file)

        loss, acc = self.evaluate(x_train2, y_train2)
        print(f"accuracy: {acc*100}")
        return loss, acc
    
    def make_prediction(self, input, board):
        if not self.trained:
            raise InputException("Model needs to be trained before making a prediction")
        ret = self.predict(input)
        legal_moves = generate_legal_moves(board)
        max_val = -10
        max_move = None
        for i in range(len(ret)):
            if MOVES[i] in legal_moves:
                if ret[i] > max_val:
                    max_val = ret[i]
                    max_move = MOVES[i]
        return max_move
    
    def save(self, tofile):
        self.save_weights(tofile)
    
    def load(self, fromfile):
        self.load_weights(fromfile)
        self.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=1e-3), loss='categorical_crossentropy', metrics="accuracy")

                

class ResidualBlock(keras.Model):
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(num_channels, padding='same', kernel_size=3, strides=strides)
        self.conv2 = tf.keras.layers.Conv2D(num_channels, kernel_size=3, padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(num_channels, kernel_size=1, strides=strides)
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.batch2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.batch1(self.conv1(X)))
        Y = self.batch2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)

input_shape = (300000, 8,8,53) # Board size for AlphaZero Chess
num_actions = OUTPUT_SIZE # Total number of possible moves on the board
num_filters = 256 # Number of filters in convolutional layers
num_residual_blocks = 9 # Number of residual blocks in the network

policy_net = PolicyNetwork(input_shape, num_actions, num_filters, num_residual_blocks)

# Compile the model 

policy_net.summary()

"""
x_train2 = np.load("data/conv3.npy")
y_train2 = np.load("data/conv4.npy")
x_train2 = x_train2.astype(np.float32)
y_train2 = y_train2.astype(np.float32)
"""
policy_net.load("models2/resNetModel")
policy_net.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=1e-3), loss='categorical_crossentropy', metrics="accuracy")
for i in range(10):
    xfilename = "data/x_train_" + str(i) + ".npy"
    yfilename = "data/y_train_" + str(i) + ".npy"
    x_train = np.load(xfilename)
    y_train = np.load(yfilename)
    policy_net.fit(x_train, y_train, epochs=1, batch_size=32)
    policy_net.save_weights("models2/resNetModel")

"""
loss, acc = policy_net.evaluate(x_train2, y_train2)
print(f"accuracy: {acc*100}")
"""  