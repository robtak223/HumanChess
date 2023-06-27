import chess.pgn
import sys

"""
Script to take a lichess database pgn file and clean the file for games with around 1500 elo and adequate time controls
"""
def clean_inputs(infile, outfile):
    with open(infile, 'r') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            try:
                white_elo = int(game.headers['WhiteElo'])
                black_elo = int(game.headers['BlackElo'])
            except:
                continue
            # only take games with rating around 1500
            if white_elo > 1400 and white_elo < 1600 and black_elo > 1400 and black_elo < 1600:
                tc = game.headers["TimeControl"]
                if tc != '-':
                    time_segments = tc.split('+')
                    base_seconds = int(time_segments[0])
                    #only take games which had more than 3 minutes
                    if base_seconds >= 180:
                        print(game, file=open(outfile, "a"), end="\n\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:\npython clean.py infile outfile")
        sys.exit(1)
    inf = sys.argv[1]
    outf = sys.argv[2]
    clean_inputs(inf, outf)
