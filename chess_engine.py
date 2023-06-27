from utils import *
from error import *

class PolicyNetwork(keras.Model):
    def __init__(self, input_shape, num_actions, num_filters, num_residual_blocks):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(num_filters, kernel_size=3, activation='relu', padding="same")
        self.res_blocks = []
        for _ in range(num_residual_blocks):
            self.res_blocks.append(ResidualBlock(num_filters))
        self.policy_conv = tf.keras.layers.Conv2D(num_filters, kernel_size=1, padding="same", activation="relu")
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
        if type(x_train[0]) == str:
            for i in range(len(x_train)):
                print(x_train[i], y_train[i])
                x_arr = np.load(x_train[i])
                y_arr = np.load(y_train[i])
                self.fit(x_arr, y_arr, epochs=epochs, batch_size=batch_size)
                self.save_weights(weights_save_file)
        self.trained = True
    
    def evaluate_model_from_file(self, x_test_file, y_test_file):
        x_train2 = np.load(x_test_file)
        y_train2 = np.load(y_test_file)
        self.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=1e-3), loss='categorical_crossentropy', metrics="accuracy")

        loss, acc = self.evaluate(x_train2, y_train2)
        print(f"accuracy: {acc*100}")
        return loss, acc
    
    def evaluate_model_from_pgn(self, infile, num):
        x_test, y_test = format_train_input(infile, num)
        self.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=1e-3), loss='categorical_crossentropy', metrics="accuracy")
        loss, acc = self.evaluate(x_test, y_test)
        print(f"accuracy: {acc*100}")
        return loss, acc
    
    def make_prediction(self, input, board):
        ret = self.predict(input)
        legal_moves = generate_legal_moves(board)
        max_val = -10
        max_move = None
        for i in range(len(ret[0])):
            if MOVES[i] in legal_moves:
                if ret[0][i] > max_val:
                    max_val = ret[0][i]
                    max_move = MOVES[i]
        return max_move
    
    def move_match_eval(self, xinputs, yinputs):
        ret = self.predict(xinputs)
        tot = 0
        for i in range(len(ret)):
            g = np.max(ret[i])
            if np.argmax(ret[i]) == g:
                tot += 1
        return float(tot) / float(len(ret))

    
    def save(self, tofile):
        self.save_weights(tofile)
    
    def load(self, fromfile):
        self.load_weights(fromfile)

                

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



