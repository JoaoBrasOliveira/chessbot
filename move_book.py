import pandas as pd
import chess
import re

class MoveBook:
    """
    MoveBook class that provides a repository of chess opening sequences and moves from Magnus Carlsen's games based on the agent's color.
    It handles opening book lookups and processing of ECO (Encyclopedia of Chess Openings) codes.

    Attributes
    ----------
    moves_white: DataFrame containing move sequences from white's perspective
    moves_black: DataFrame containing move sequences from black's perspective
    openings: Dictionary of chess openings converted from ECO codes

    Methods
    ----------
    __init__()
        Initializes the MoveBook with dataframes for white moves, black moves, and openings
    _convert_ecocodes_to_dict()
        Converts ECO openings DataFrame into a structured dictionary
    get_opening_moves()
        Retrieves opening moves for a given ECO code and player color
    get_magnus_moves()
        Returns moves from a randomly chosen Magnus Carlsen game based on player color
    get_response_to_position()
        Attempts to find a move that responds to the current board position

    """
    def __init__(self, moves_white: pd.DataFrame, moves_black: pd.DataFrame, openings: pd.DataFrame):
        self.moves_white = moves_white
        self.moves_black = moves_black
        self.openings = self._convert_ecocodes_to_dict(openings)

    def _convert_ecocodes_to_dict(self, df_ecocodes):
        """
        Converts the ECO openings DataFrame into a dictionary keyed by ECO code.
        Each value is a dict with keys 'white' and 'black' holding the respective moves.
        """
        openings_dict = {}
        for _, row in df_ecocodes.iterrows():
            eco_code = row["eco"]  # Use the ECO code as the key.
            move_sequence = row["eco_example"]

            # Split the string into tokens using whitespace.
            tokens = re.split(r'\s+', move_sequence.strip())
            moves = []
            for token in tokens:
                # Remove common trailing punctuation (like commas or periods).
                token = token.strip(".,")
                # Skip tokens that are just move numbers or extraneous words.
                if token.isdigit():
                    continue
                if token.lower() in ["etc", "etc."]:
                    continue
                moves.append(token)

            # Assume moves are alternating: White's moves at even indices and Black's at odd.
            white_moves = moves[::2]
            black_moves = moves[1::2]
            openings_dict[eco_code] = {"white": white_moves, "black": black_moves}
        return openings_dict

    def get_opening_moves(self, opening_code: str, color: bool = chess.WHITE) -> list:
        """
        Retrieve the opening moves for a given ECO code and color.
        Returns the list of moves if found; otherwise, returns an empty list.
        """
        opening = self.openings.get(opening_code, {})
        return opening.get("white", []) if color == chess.WHITE else opening.get("black", [])

    def get_magnus_moves(self, color: bool) -> list:
        """
        Returns a list of moves (in SAN notation) for a randomly chosen game
        from Magnus Carlsen’s games. For white, it takes even–indexed moves;
        for black, odd–indexed moves.

        Returns an empty list if there's an error parsing the move sequence.
        """
        try: # Added "try" for error handling.
            if color == chess.WHITE:
                chosen_row = self.moves_white.sample(n=1).iloc[0]
            else:
                chosen_row = self.moves_black.sample(n=1).iloc[0]
            move_sequence = chosen_row['move_sequence']
            if '|' in move_sequence:
                full_moves = move_sequence.split('|')
            else:
                full_moves = move_sequence.split()
            return full_moves[0::2] if color == chess.WHITE else full_moves[1::2]
        except Exception as e: # Added exception for error handling.
            print(f"Error processing Magnus game: {e}")
            return []  # Return empty list on error

def get_response_to_position(self, board: chess.Board, color: bool) -> chess.Move: #Added to attempt to fix errors
        """
        New method: Try to find a move from the database that responds to the current position.
        This is a fallback when direct move sequence doesn't work.
        """
        try:
            # Sample multiple games to increase chances of finding a matching position
            sample_size = min(20, len(self.moves_white if color == chess.WHITE else self.moves_black))
            sample_rows = (self.moves_white if color == chess.WHITE else self.moves_black).sample(n=sample_size)
            
            # Try each game in the sample
            for _, row in sample_rows.iterrows():
                move_sequence = row['move_sequence']
                if '|' in move_sequence:
                    full_moves = move_sequence.split('|')
                else:
                    full_moves = move_sequence.split()
                
                # Set up a test board to find positions that match our current board
                test_board = chess.Board()
                for i, move_san in enumerate(full_moves):
                    try:
                        move = test_board.parse_san(move_san)
                        test_board.push(move)
                        
                        # If we find a matching position and it's our turn
                        if test_board.board_fen() == board.board_fen() and test_board.turn == color:
                            # Get the next move from this sequence if available
                            if i + 1 < len(full_moves):
                                next_move_san = full_moves[i + 1]
                                move = board.parse_san(next_move_san)
                                if move in board.legal_moves:
                                    return move
                    except Exception:
                        # Skip errors and continue trying moves
                        break
            
            return None  # No matching position found
        except Exception as e:
            print(f"Error in get_response_to_position: {e}")
            return None  # Return None on error