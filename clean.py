import chess.pgn
import sys

"""
Script to take a lichess database pgn file and clean the file for games with around 1500 elo and adequate time controls
"""
def clean_inputs(infile, outfiles):
    f1 = open(outfiles[0], "a")
    f2 = open(outfiles[1], "a")
    f3 = open(outfiles[2], "a")
    f4 = open(outfiles[3], "a")
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
            tc = game.headers["TimeControl"]
            if tc != '-':
                time_segments = tc.split('+')
                base_seconds = int(time_segments[0])
                #only take games which had more than 3 minutes
                if base_seconds >= 180:
                    # only take games with rating around 1500
                    if white_elo > 1000 and white_elo < 1200 and black_elo > 1000 and black_elo < 1200:
                        print(game, file=f1, end="\n\n")
                    elif white_elo > 1200 and white_elo < 1400 and black_elo > 1200 and black_elo < 1400:
                        print(game, file=f2, end="\n\n")
                    elif white_elo > 1600 and white_elo < 1800 and black_elo > 1600 and black_elo < 1800:
                        print(game, file=f3, end="\n\n")
                    elif white_elo > 1800 and white_elo < 2000 and black_elo > 1800 and black_elo < 2000:
                        print(game, file=f4, end="\n\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:\npython clean.py infile")
        sys.exit(1)
    inf = sys.argv[1]
    outf = ["data/1100.pgn", "data/1300.pgn", "data/1700.pgn", "data/1900.pgn"]
    clean_inputs(inf, outf)
