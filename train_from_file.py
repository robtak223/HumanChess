from chess_engine import *
from common import *
import sys

server_net = PolicyNetwork(INPUT_SHAPE, OUTPUT_SIZE, NUM_FILTERS, NUM_RES_BLOCKS)
#server_net.load("models/model1/resNetModel")
xfiles = []
yfiles = []
server_net.load("models/model4/meModel")
for i in range(1):
    xf = "arrays/xp_0" + ".npy"
    yf = "arrays/yp_0" + ".npy"
    xfiles.append(xf)
    yfiles.append(yf)
#server_net.train_model(xfiles, yfiles, "models/model9/resNetModel", epochs=1, lr=1e-5)
xf = "arrays/x_" + str(16) + ".npy"
yf = "arrays/y_" + str(16) + ".npy"
loss, acc = server_net.evaluate_model_from_file(xf, yf)
print(loss, acc)
