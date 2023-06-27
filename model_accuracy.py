from chess_engine import *
import sys

def main(test_file, model_file, num):
    server_net = PolicyNetwork(INPUT_SHAPE, OUTPUT_SIZE, NUM_FILTERS, NUM_RES_BLOCKS)
    server_net.load(model_file)
    loss, acc = server_net.evaluate_model_from_pgn(test_file, num)


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage:\npython model_accuracy.py test_file model_file number_samples")
        sys.exit(1)
    inf = sys.argv[1]
    modelw = sys.argv[2]
    num = sys.argv[3]
    main(inf, modelw, num)