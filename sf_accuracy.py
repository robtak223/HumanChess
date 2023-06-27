from stockfish import Stockfish
import chess.pgn
import numpy as np
import sys

def main(file, num):
    
    
    stockfish = Stockfish(path="/opt/homebrew/Cellar/stockfish/15.1/bin/stockfish")
    j = 2
    while j < 80:
        f = open(file)
        tot = 0
        numdo = 0
        i = 0
        while i < num:
            game = chess.pgn.read_game(f)
            if game == None:
                break
            length = game.end().ply()
            if length <= j+2:
                continue
            while game.ply() < j:
                game = game.next()
            game2 = game.next()
            move = game2.move.uci()
            fen = game.board().fen()
            stockfish.set_fen_position(fen)
            sf_move = stockfish.get_best_move()
            if sf_move == move:
                tot += 1
            i+=1
        print(tot, i)
        print(f"accuracy for move : {round(100 * float(tot) / float(), 2)}%")
        j+=5

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:\npython sf_accuracy.py infile num")
        sys.exit(1)
    inf = sys.argv[1]
    num = int(sys.argv[2])
    main(inf, num)


    