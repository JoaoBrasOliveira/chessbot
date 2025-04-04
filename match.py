import chess
import chess.svg
import time
from IPython.display import clear_output, SVG, display
from move_book import MoveBook
from chess_agent import ChessAgent
import pandas as pd
import sys
import os
import unittest
from tests import TestChessAgent, TestMoveBook, TestMatch # Import test cases

class Match:
    """
    Match class manages a chess match between two agents, including scheduling moves, tracking time, and rendering the game state.

    Parameters
    ----------
    white_agent: The ChessAgent playing white
    black_agent: The ChessAgent playing black
    board: The chess.Board object representing the current game state
    white_clock: Time remaining for the white player (in seconds)
    black_clock: Time remaining for the black player (in seconds)

    Methods
    ----------
    __init__()
        Initializes a match between two agents
    render()
        Displays the current board state using SVG
    play()
        Runs the match until completion, managing turns and time controls

    """
    def __init__(self, white_agent: ChessAgent, black_agent: ChessAgent):
        self.white_agent = white_agent
        self.black_agent = black_agent
        self.board = chess.Board()
        self.white_clock = 10.0  # total seconds available to White
        self.black_clock = 10.0  # total seconds available to Black

    def render(self):
        # Render the board as SVG for Kaggle Environment Rendering.
        clear_output(wait=True)
        display(SVG(chess.svg.board(board=self.board)))
        print("White clock: {:.2f}s, Black clock: {:.2f}s".format(
            self.white_clock, self.black_clock))
    
    def play(self):
        while not self.board.is_game_over():
            # Determine which agent is to move and get their remaining time.
            current_agent = self.white_agent if self.board.turn == chess.WHITE else self.black_agent
            available_time = self.white_clock if self.board.turn == chess.WHITE else self.black_clock
            
            # If a player's time is exhausted, they forfeit on time.
            if available_time <= 0:
                print(f"{'White' if self.board.turn == chess.WHITE else 'Black'} has run out of time!")
                break
            
            # Record start time for this move.
            move_start = time.time()
            move = current_agent.select_move(self.board, time_limit=available_time)
            move_time = time.time() - move_start
            
            # Update the clock: subtract elapsed time and add a fixed 0.1s increment.
            if self.board.turn == chess.WHITE:
                self.white_clock = max(0, self.white_clock - move_time) + 0.1
            else:
                self.black_clock = max(0, self.black_clock - move_time) + 0.1
            self.board.push(move)
            self.render()
        result = self.board.result(claim_draw=True)
        print("Game Over:", result)
        return result
    
# ------------------------------------------------------------------------------
# Main Execution Block
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    
    if '--test' in sys.argv: #python match.py --test
        # Run tests if the '--test' flag is provided
        print("Running tests...")
        unittest.main(argv=[''], exit=False)
    
    else:
        # Read the gzip-compressed files
        
        # Define the path to the "data" folder relative to the script's directory
        data_folder = os.path.join(os.path.dirname(__file__), 'data')

        moves_black = pd.read_csv(os.path.join(data_folder, 'black.csv.gz'), compression='gzip')
        moves_white = pd.read_csv(os.path.join(data_folder, 'white.csv.gz'), compression='gzip')
        openings = pd.read_csv(os.path.join(data_folder, 'openings.csv.gz'), compression='gzip')

        # Define the test opening code you wish to use (must match the ECO code, e.g. "A01")
        test_opening_code = "C34" # King's Gambit Accepted

        # Initialize the opening book with the appropriate dataframes.
        move_book = MoveBook(moves_white, moves_black, openings)

        # Create agents – for example, the White agent uses the test opening moves.
        white_agent = ChessAgent(chess.WHITE, move_book, test_opening_code=test_opening_code)
        black_agent = ChessAgent(chess.BLACK, move_book)  # Black could use Magnus’ moves

        # Play Game
        match = Match(white_agent, black_agent)
        match.play()