#pip install python-chess

import chess
import chess.svg
import time
import random
import pandas as pd
from IPython.display import clear_output, SVG, display
import re

# Read the gzip-compressed CSV file
moves_black = pd.read_csv('black.csv.gz', compression='gzip')
moves_white = pd.read_csv('white.csv.gz', compression='gzip')
openings = pd.read_csv('openings.csv.gz', compression='gzip')

# ------------------------------------------------------------------------------
# MoveBook: Provides a list of opening names and move sequences as well as moves from Magnus Carlsen games based on the agent's color.
# ------------------------------------------------------------------------------

class MoveBook:
    """
    Provides a repository of chess opening sequences and moves from Magnus Carlsen's games based on the agent's color.
    It handles opening book lookups and processing of ECO (Encyclopedia of Chess Openings) codes.

    Parameters
    ----------
    moves_white: DataFrame containing move sequences from white's perspective
    moves_black: DataFrame containing move sequences from black's perspective
    openings: Dictionary of chess openings converted from ECO codes

    

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
        """
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

# ------------------------------------------------------------------------------
# ChessAgent: Advanced agent that uses an opening book then an iterative deepening
# search with alpha–beta pruning. It obeys a 0.1s delay and 10s move limit.
# ------------------------------------------------------------------------------

class ChessAgent:
    def __init__(self, color: bool, move_book: MoveBook = None, test_opening_code: str = None):
        self.color = color
        self.move_book = move_book

        # Retrieve test opening moves (for the given color) if a test code is provided.
        self.test_opening_moves = move_book.get_opening_moves(test_opening_code, color) if test_opening_code else []
        self.magnus_moves = move_book.get_magnus_moves(color)
        self.opening_moves_played = 0  # Track moves played from the test opening
        self.magnus_moves_played = 0   # Track moves played from Magnus’ games


    def select_move(self, board: chess.Board, time_limit: float = 10.0) -> chess.Move:
        # Fixed 0.1 second delay.
        time.sleep(0.1)

        # 🔹 Test Mode: Use test opening moves (if provided and valid).
        if self.color == chess.WHITE and self.test_opening_moves:
            if self.opening_moves_played < len(self.test_opening_moves):
                expected_move = self.test_opening_moves[self.opening_moves_played]
                try:
                    move = board.parse_san(expected_move)
                    if move in board.legal_moves:
                        print(f"Playing {expected_move} ({self.opening_moves_played + 1}/{len(self.test_opening_moves)})")
                        self.opening_moves_played += 1
                        return move
                except Exception as e:
                    print(f"Error in test opening move: {e}")

        # 🔹 Use Magnus' moves if available.
        elif self.magnus_moves_played < len(self.magnus_moves):
            expected_move = self.magnus_moves[self.magnus_moves_played]
            try:
                move = board.parse_san(expected_move)
                if move in board.legal_moves:
                    print(f"Playing {expected_move} ({self.magnus_moves_played + 1}/{len(self.magnus_moves)})")
                    self.magnus_moves_played += 1
                    return move
            except Exception as e:
                print(f"Magnus move error: {e}")

        # 🔹 Otherwise, use Alpha-Beta Search with iterative deepening.
        start_time = time.time()
        best_move = None
        depth = 1
        while time.time() - start_time < time_limit:
            move, _ = self.alpha_beta_search(board, depth, -float('inf'), float('inf'), self.color, start_time, time_limit)
            if move is not None:
                best_move = move
            depth += 1

        # Instead of falling back to a random move,
        # evaluate all legal moves and pick the one with the best evaluation.
        if best_move is not None:
            return best_move
        else:
            fallback_move = None
            best_eval = -float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_val = self.evaluate(board)
                board.pop()
                if eval_val > best_eval:
                    best_eval = eval_val
                    fallback_move = move
            return fallback_move

    def alpha_beta_search(self, board: chess.Board, depth: int, alpha: float, beta: float,
                          player: bool, start_time: float, time_limit: float):
        # Check termination conditions.
        if depth == 0 or board.is_game_over() or (time.time() - start_time) > time_limit:
            return None, self.evaluate(board)
        best_move = None
        if board.turn == player:
            value = -float('inf')
            for move in board.legal_moves:
                board.push(move)
                _, score = self.alpha_beta_search(board, depth - 1, alpha, beta,
                                                  player, start_time, time_limit)
                board.pop()
                if score > value:
                    value = score
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # Beta cutoff.
            return best_move, value
        else:
            value = float('inf')
            for move in board.legal_moves:
                board.push(move)
                _, score = self.alpha_beta_search(board, depth - 1, alpha, beta,
                                                  player, start_time, time_limit)
                board.pop()
                if score < value:
                    value = score
                    best_move = move
                beta = min(beta, value)
                if alpha >= beta:
                    break  # Alpha cutoff.
            return best_move, value

    def evaluate(self, board: chess.Board, depth: int = 0) -> float:
        """
        Enhanced evaluation function that:
        - Returns extreme values for mate/stalemate.
        - Combines material count, mobility, piece safety, and central control.
        - Rewards advanced passed pawns.
        - Discourages non-profitable captures.
        - And—critically—if the opponent has only the king remaining (and we have mating material),
          it rewards moves that confine the enemy king (reducing its mobility and forcing it to the edge)
          and delivers checks.
        Additionally, this version incorporates:
        - Threats and counter-threats,
        - Multiple piece coordination,
        - Penalties for positions that do not show positional improvement,
        - And a penalty for disruptive captures when in a dominant position.
        """
        # Terminal states.
        if board.is_checkmate():
            return 10000 - depth if board.turn != self.color else -10000 + depth
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        # Basic material values.
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        score = 0

        # --- Material Count ---
        for piece_type, value in piece_values.items():
            score += len(board.pieces(piece_type, self.color)) * value
            score -= len(board.pieces(piece_type, not self.color)) * value

        # --- Passed Pawn Evaluation ---
        def is_passed_pawn(square, color):
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            if color == chess.WHITE:
                for r in range(rank+1, 8):
                    for f in range(max(0, file-1), min(7, file+1)+1):
                        sq = chess.square(f, r)
                        piece = board.piece_at(sq)
                        if piece is not None and piece.piece_type == chess.PAWN and piece.color != color:
                            return False
                return True
            else:
                for r in range(rank-1, -1, -1):
                    for f in range(max(0, file-1), min(7, file+1)+1):
                        sq = chess.square(f, r)
                        piece = board.piece_at(sq)
                        if piece is not None and piece.piece_type == chess.PAWN and piece.color != color:
                            return False
                return True

        def open_road(square, color):
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            if color == chess.WHITE:
                for r in range(rank+1, 8):
                    sq = chess.square(file, r)
                    if board.piece_at(sq) is not None:
                        return False
                return True
            else:
                for r in range(rank-1, -1, -1):
                    sq = chess.square(file, r)
                    if board.piece_at(sq) is not None:
                        return False
                return True

        # Reward our advanced passed pawns.
        for pawn in board.pieces(chess.PAWN, self.color):
            rank = chess.square_rank(pawn)
            if is_passed_pawn(pawn, self.color):
                bonus = (rank * 0.5) if self.color == chess.WHITE else ((7 - rank) * 0.5)
                if open_road(pawn, self.color):
                    bonus += 2.0  # Extra bonus for a completely open road.
                score += bonus

        # Penalize enemy passed pawns.
        for pawn in board.pieces(chess.PAWN, not self.color):
            rank = chess.square_rank(pawn)
            if is_passed_pawn(pawn, not self.color):
                bonus = ((7 - rank) * 0.5) if self.color == chess.WHITE else (rank * 0.5)
                if open_road(pawn, not self.color):
                    bonus += 2.0
                score -= bonus

        # --- Threats and Counter-Threats Bonus ---
        # For each of our pieces that is attacked, add a bonus if we also threaten a more valuable enemy piece.
        counter_threat_bonus = 0
        for square, piece in board.piece_map().items():
            if piece.color == self.color:
                if board.attackers(not self.color, square):
                    threatened_value = piece_values[piece.piece_type]
                    for enemy_sq, enemy_piece in board.piece_map().items():
                        if enemy_piece.color != self.color and board.is_attacked_by(self.color, enemy_sq):
                            enemy_value = piece_values[enemy_piece.piece_type]
                            if enemy_value > threatened_value:
                                counter_threat_bonus += (enemy_value - threatened_value) * 0.2
        score += counter_threat_bonus

        # --------------------------
        # Advanced Piece Safety Evaluation
        # --------------------------
        # For every non-king piece, calculate a safety margin defined as:
        #     (number of defenders) - (number of attackers)
        # If the margin is negative, the piece is considered "hanging."
        # We apply a penalty proportional to the imbalance and to the piece's material value.
        safety_factor = 0.5  # Adjust to tune the penalty severity.
        extra_hanging_penalty = 0.5  # Additional penalty if attacked by 2+ enemy pieces with no defense.
        for color in [self.color, not self.color]:
            # For our pieces, a negative margin lowers our score; for opponent's pieces it raises ours.
            sign = 1 if color == self.color else -1
            for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                for square in board.pieces(piece_type, color):
                    attackers = board.attackers(not color, square)
                    defenders = board.attackers(color, square)
                    safety_margin = len(defenders) - len(attackers)
                    if safety_margin < 0:
                        # The penalty increases with both the material value and the deficit.
                        penalty = abs(safety_margin) * piece_values[piece_type] * safety_factor
                        score -= sign * penalty
                    # If a piece is attacked by two or more enemy pieces and has no defenders,
                    # apply an extra penalty.
                    if len(attackers) >= 2 and len(defenders) == 0:
                        extra_penalty = piece_values[piece_type] * extra_hanging_penalty
                        score -= sign * extra_penalty

        # --- Reward Opportunities to Capture Underdefended Opponent Pieces ---
        for piece_type, value in piece_values.items():
            if piece_type == chess.KING:
                continue
            for square in board.pieces(piece_type, not self.color):
                attackers = board.attackers(self.color, square)
                defenders = board.attackers(not self.color, square)
                if len(attackers) > len(defenders):
                    score += value * 0.3

        # --- Mobility Bonus ---
        def mobility(b, color):
            original_turn = b.turn
            b.turn = color
            moves_count = len(list(b.legal_moves))
            b.turn = original_turn
            return moves_count

        mobility_factor = 0.05
        score += mobility_factor * (mobility(board, self.color) - mobility(board, not self.color))

        # --- Central Control Bonus ---
        central_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        central_control_bonus = 0.2
        for square in central_squares:
            if board.is_attacked_by(self.color, square):
                score += central_control_bonus
            if board.is_attacked_by(not self.color, square):
                score -= central_control_bonus

        # --- Piece Coordination Bonus ---
        # Reward positions where our pieces support one another.
        coordination_bonus = 0
        for square, piece in board.piece_map().items():
            if piece.color == self.color:
                defenders = board.attackers(self.color, square)
                if len(defenders) > 1:
                    coordination_bonus += (len(defenders) - 1) * 0.2
        score += coordination_bonus

        # --- Discourage Non-Positive Captures ---
        # If our last move was a capture that did not result in a net material gain, apply a small penalty.
        if board.move_stack:
            last_move = board.peek()
            if board.turn != self.color and board.is_capture(last_move):
                material_diff = 0
                for piece_type, value in piece_values.items():
                    material_diff += len(board.pieces(piece_type, self.color)) * value
                    material_diff -= len(board.pieces(piece_type, not self.color)) * value
                if material_diff < 1:  # No net material gain
                    score -= 1.0

        # --- Positional Improvement Check ---
        # Combine non-material factors: mobility, central control, and coordination.
        my_mobility = mobility(board, self.color)
        opp_mobility = mobility(board, not self.color)
        pos_metric = mobility_factor * (my_mobility - opp_mobility)
        central_metric = 0
        for square in central_squares:
            if board.is_attacked_by(self.color, square):
                central_metric += central_control_bonus
            if board.is_attacked_by(not self.color, square):
                central_metric -= central_control_bonus
        pos_metric += central_metric
        pos_metric += coordination_bonus
        # Penalize positions that do not show a measurable improvement.
        if pos_metric < 0.5:
            score -= 2.0

        # --- Dominant Position Capture Penalty ---
        # When our non-material position is dominant, a capture that may disrupt it is discouraged.
        dominance_threshold = 3.0
        if pos_metric > dominance_threshold:
            if board.move_stack:
                last_move = board.peek()
                if board.is_capture(last_move):
                    score -= 2.0

        # --- Forced Mate Heuristic ---
        # Reward positions that confine the enemy king.
        if len(board.pieces(chess.KING, not self.color)) == 1:
            enemy_king_sq = board.king(not self.color)
            enemy_king_moves = sum(1 for move in board.legal_moves if move.from_square == enemy_king_sq)
            mate_bonus = (10 - enemy_king_moves) * 50  # Fewer moves = higher bonus.
            score += mate_bonus

            enemy_king_file = chess.square_file(enemy_king_sq)
            enemy_king_rank = chess.square_rank(enemy_king_sq)
            if enemy_king_file in [0, 7] or enemy_king_rank in [0, 7]:
                score += 100

            if board.is_check():
                if board.turn == self.color:
                    score += 20 # If it's our turn and we are giving check, add a bonus
                else:
                    score -= 20 # Otherwise, if our king is in check, subtract.
        return score

# ------------------------------------------------------------------------------
# Match: Schedules and renders a game between two agents.
# ------------------------------------------------------------------------------

class Match:
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

    def play(self) -> str:
        while not self.board.is_game_over():
            # Determine which agent is to move and get their remaining time.
            if self.board.turn == chess.WHITE:
                current_agent = self.white_agent
                available_time = self.white_clock
                player_label = 'White'
            else:
                current_agent = self.black_agent
                available_time = self.black_clock
                player_label = 'Black'

            # If a player's time is exhausted, they forfeit on time.
            if available_time <= 0:
                print(f"{player_label} has run out of time!")
                break

            # Record start time for this move.
            move_start = time.time()
            move = current_agent.select_move(self.board, time_limit=available_time)
            move_time = time.time() - move_start

            # Update the clock: subtract elapsed time and add a fixed 0.1s increment.
            if player_label == 'White':
                self.white_clock = max(0, self.white_clock - move_time)
                self.white_clock = min(10.0, self.white_clock + 0.1)  # do not exceed 10s
            else:
                self.black_clock = max(0, self.black_clock - move_time)
                self.black_clock = min(10.0, self.black_clock + 0.1)

            self.board.push(move)
            self.render()

        result = self.board.result(claim_draw=True)
        print("Game Over:", result)
        return result

# ------------------------------------------------------------------------------
# Main Execution (Run the Game)
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    
    # Define the test opening code you wish to use (must match the ECO code, e.g. "A01")
    test_opening_code = "A01"

    # Initialize the opening book with the appropriate dataframes.
    move_book = MoveBook(moves_white, moves_black, openings)

    # Create agents – for example, the White agent uses the test opening moves.
    white_agent = ChessAgent(chess.WHITE, move_book, test_opening_code=test_opening_code)
    black_agent = ChessAgent(chess.BLACK, move_book)  # Black could use Magnus’ moves

    # Play Game with Kaggle Rendering
    match = Match(white_agent, black_agent)
    match.play()