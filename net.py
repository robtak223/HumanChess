from chess_engine import *
from common import *

server_net_gen = PolicyNetwork(INPUT_SHAPE, OUTPUT_SIZE, NUM_FILTERS, NUM_RES_BLOCKS)
server_net_gen.load("models/model9/resNetModel")
server_net_per = PolicyNetwork(INPUT_SHAPE, OUTPUT_SIZE, NUM_FILTERS, NUM_RES_BLOCKS)
server_net_per.load("models/model4/meModel")