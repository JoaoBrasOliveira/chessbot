import chess
import time
from move_book import MoveBook

class ChessAgent:
    """
    ChessAgent class provides Advanced chess-playing agent that uses opening books and iterative deepening search with alpha-beta pruning.
    It obeys time constraints with a 0.1s delay and 10s move limit.

    Attributes
    ----------
    color: Boolean indicating the agent's color (True for white, False for black)
    move_book: Reference to the MoveBook object for opening moves
    test_opening_moves: List of moves from a test opening
    magnus_moves: List of moves from Magnus Carlsen's games
    opening_moves_played: Counter tracking moves played from the test opening
    magnus_moves_played: Counter tracking moves played from Magnus' games

    Methods
    ----------
    __init__()
        Initializes the agent with a color and optional move book
    select_move()
        Selects the best move using prioritized strategies
    alpha_beta_search()
        Performs alpha-beta pruning search
    evaluate()
        Evaluates board positions
    is_move_legal()
        New helper method to check if a move is legal

    """
    def __init__(self, color: bool, move_book: MoveBook = None, test_opening_code: str = None):
        self.color = color
        self.move_book = move_book

        # Retrieve test opening moves (for the given color) if a test code is provided.
        self.test_opening_moves = move_book.get_opening_moves(test_opening_code, color) if test_opening_code else []
        self.magnus_moves = move_book.get_magnus_moves(color)
        self.opening_moves_played = 0  # Track moves played from the test opening
        self.magnus_moves_played = 0   # Track moves played from Magnus’ games
        self.failed_moves_count = 0    # Track consecutive failed moves to avoid repeated errors - added for error handling

    def is_move_legal(self, board: chess.Board, move_san: str) -> chess.Move: # Added for error handling
            """Helper method to check if a move in SAN notation is legal and return the Move object"""
            try:
                move = board.parse_san(move_san)
                if move in board.legal_moves:
                    return move
                return None
            except Exception:
                return None

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