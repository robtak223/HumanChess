import numpy as np
import chess.pgn
from common import *

def bits_from_board(game):
    board = game.board()
    fen = board.fen()
    fields = fen.split(' ')
    bits = [0 for i in range(INPUT_SIZE)]
    fen_board = fields[0]
    pos = 0
    for char in fen_board:
        if char.isdigit():
            pos += (NUM_PIECES * int(char))
        elif char.isalpha():
            bits[pos+POSITIONS[char]] = 1
            pos += NUM_PIECES
    if fields[1] == 'w':
        bits[TURN_BIT] = 1
    if 'K' in fields[2]:
        bits[WK_CASTLE_BIT] = 1
    if 'Q' in fields[2]:
        bits[WQ_CASTLE_BIT] = 1
    if 'k' in fields[2]:
        bits[BK_CASTLE_BIT] = 1
    if 'q' in fields[2]:
        bits[BQ_CASTLE_BIT] = 1
    return bits
         
def prepare_train(infile, num, x_train, y_train, true_move):
    try:
        f = open(infile)
    except:
        print(f"File name: {infile} does not exist")
        raise Exception
    for i in range(num):
        game = chess.pgn.read_game(f)
        length = game.end().ply()
        if length <= 15:
            continue
        one = np.random.randint(10, length-5)
        two = np.random.randint(10, length-5)           
        while game.ply() < one and game.ply() < two:
            game = game.next()
        game1 = game
        while game.ply() < one or game.ply() < two:
            game = game.next()
        game2 = game
        for game in [game1, game2]:
            x_train.append(np.array(bits_from_board(game)))
            next_move = MOVES_INDEX[game.next().move.uci()]
            true_move[next_move] = 1
            y_train.append(np.array(true_move))
            true_move[next_move] = 0


def prepare_test(infile, num, x_train2, y_train2, true_move):
    try:
        f = open(infile)
    except:
        print(f"File name: {infile} does not exist")
        raise Exception
    for i in range(num):
        game = chess.pgn.read_game(f)
        length = game.end().ply()
        if length <= 15:
            continue
        one = np.random.randint(10, length-5)
        two = np.random.randint(10, length-5)          
        while game.ply() < one and game.ply() < two:
            game = game.next()
        game1 = game
        while game.ply() < one or game.ply() < two:
            game = game.next()
        game2 = game
        for game in [game1, game2]:
            x_train2.append(np.array(bits_from_board(game)))
            next_move = MOVES_INDEX[game.next().move.uci()]
            true_move[next_move] = 1
            y_train2.append(np.array(true_move))
            true_move[next_move] = 0

def format_train_input(infile, num):
    try:
        f = open(infile)
    except:
        print(f"File name: {infile} does not exist")
        raise Exception
    outputs = []
    inputs = []
    count = 0
    for i in range(num):
        count += 1
        if not count % 10000:
            print(count // 10000)
        game = chess.pgn.read_game(f)
        length = game.end().ply()
        if length <= 20:
            continue
        one = np.random.randint(1, length-1)
        two = np.random.randint(1, length-1)
        temp =one
        one = min(one, two)  
        two = max(two, temp)        
        while game.ply() < one-NUM_PREV_POSITIONS:
            game = game.next()
        game1 = game
        while game.ply() < one-NUM_PREV_POSITIONS or game.ply() < two-NUM_PREV_POSITIONS:
            game = game.next()
        game2 = game
        mark = one
        for game in [game1, game2]:
            fields = None
            if game == game2:
                mark = two
            true_move = np.array([0 for i in range(OUTPUT_SIZE)])
            new_sample = np.array([np.array([np.array([0 for p in range(BQ_CASTLE_CHANNEL+1)]) for k in range(8)]) for j in range(8)])
            while game.ply() <= mark:
                board = game.board()
                fen = board.fen()
                fields = fen.split(' ')
                fen_board = fields[0]
                pos = 0
                for char in fen_board:
                    if char.isdigit():
                        pos += int(char)
                    elif char.isalpha():
                        pos1 = pos // 8
                        pos2 = pos % 8
                        new_sample[pos1][pos2][(mark-game.ply())*NUM_PIECES + POSITIONS[char]] = 1
                        pos += 1
                game = game.next()
            for p in range(8):
                for k in range(8):
                    new_sample[p][k][TURN_CHANNEL] = int(fields[1] == 'w')
                    new_sample[p][k][WK_CASTLE_CHANNEL] = int('K' in fields[2])
                    new_sample[p][k][WQ_CASTLE_CHANNEL] = int('Q' in fields[2])
                    new_sample[p][k][BK_CASTLE_CHANNEL] = int('k' in fields[2])
                    new_sample[p][k][BQ_CASTLE_CHANNEL] = int('q' in fields[2])  
            
            next_move = MOVES_INDEX[game.next().move.uci()]
            true_move[next_move] = 1
            inputs.append(new_sample)
            outputs.append(true_move)
    return np.array(inputs).astype(np.float32), np.array(outputs).astype(np.float32)

def generate_legal_moves(board):
    ret = set()
    for move in board.legal_moves:
        ret.add(move.uci())
    return ret
