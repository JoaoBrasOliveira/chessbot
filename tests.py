import unittest
import chess
import pandas as pd
from main import MoveBook, ChessAgent, Match
from unittest.mock import patch, MagicMock

class TestMoveBook(unittest.TestCase):
    def setUp(self):
        # Create minimal test data
        self.test_moves_white = pd.DataFrame({
            'move_sequence': ['e4 e5 Nf3 Nc6 Bb5', 'd4 d5 c4 e6 Nc3']
        })
        self.test_moves_black = pd.DataFrame({
            'move_sequence': ['e4 c5 Nf3 d6 d4', 'e4 e6 d4 d5 Nc3']
        })
        self.test_openings = pd.DataFrame({
            'eco': ['A01', 'B20'],
            'name': ['Nimzovich-Larsen Attack', 'Sicilian Defense'],
            'eco_example': ['1. b3 e5 2. Bb2', '1. e4 c5']
        })
        self.move_book = MoveBook(self.test_moves_white, self.test_moves_black, self.test_openings)

    def test_convert_ecocodes_to_dict(self):
        """Test that ECO codes are correctly converted to a dictionary structure"""
        openings = self.move_book.openings
        self.assertIn('A01', openings)
        self.assertIn('B20', openings)
        self.assertEqual(openings['A01']['white'], ['b3', 'Bb2'])
        self.assertEqual(openings['A01']['black'], ['e5'])
        self.assertEqual(openings['B20']['white'], ['e4'])
        self.assertEqual(openings['B20']['black'], ['c5'])

    def test_get_opening_moves(self):
        """Test retrieving opening moves for a specific color"""
        white_moves = self.move_book.get_opening_moves('A01', chess.WHITE)
        black_moves = self.move_book.get_opening_moves('A01', chess.BLACK)
        self.assertEqual(white_moves, ['b3', 'Bb2'])
        self.assertEqual(black_moves, ['e5'])
        
    def test_get_opening_moves_nonexistent(self):
        """Test retrieving moves for a non-existent opening code"""
        moves = self.move_book.get_opening_moves('Z99', chess.WHITE)
        self.assertEqual(moves, [])
        
    def test_get_magnus_moves(self):
        """Test retrieving moves from Magnus database"""
        with patch.object(self.test_moves_white, 'sample') as mock_sample:
            mock_sample.return_value = pd.DataFrame({'move_sequence': ['e4 e5 Nf3 Nc6 Bb5']})
            white_moves = self.move_book.get_magnus_moves(chess.WHITE)
            self.assertEqual(white_moves, ['e4', 'Nf3', 'Bb5'])
            
        with patch.object(self.test_moves_black, 'sample') as mock_sample:
            mock_sample.return_value = pd.DataFrame({'move_sequence': ['e4 c5 Nf3 d6 d4']})
            black_moves = self.move_book.get_magnus_moves(chess.BLACK)
            self.assertEqual(black_moves, ['c5', 'd6'])
    
    def test_get_response_to_position(self):
        """Test finding a response to a specific board position"""
        board = chess.Board()
        board.push_san("e4")
        board.push_san("c5")
        
        with patch.object(self.test_moves_white, 'sample') as mock_sample:
            mock_df = pd.DataFrame({'move_sequence': ['e4 c5 Nf3']})
            mock_sample.return_value = mock_df
            move = self.move_book.get_response_to_position(board, chess.WHITE)
            self.assertIsNotNone(move)
            self.assertEqual(board.san(move), "Nf3")


class TestChessAgent(unittest.TestCase):
    def setUp(self):
        # Create mock move book
        self.mock_move_book = MagicMock()
        self.mock_move_book.get_opening_moves.return_value = ['e4', 'Nf3']
        self.mock_move_book.get_magnus_moves.return_value = ['d4', 'c4']
        self.white_agent = ChessAgent(chess.WHITE, self.mock_move_book, test_opening_code="A01")
        self.black_agent = ChessAgent(chess.BLACK, self.mock_move_book)
    
    def test_initialization(self):
        """Test that agent initializes with correct attributes"""
        self.assertEqual(self.white_agent.color, chess.WHITE)
        self.assertEqual(self.white_agent.test_opening_moves, ['e4', 'Nf3'])
        self.assertEqual(self.white_agent.magnus_moves, ['d4', 'c4'])
        self.assertEqual(self.white_agent.opening_moves_played, 0)
        
    def test_is_move_legal(self):
        """Test move legality checker"""
        board = chess.Board()
        # Legal move
        move = self.white_agent.is_move_legal(board, "e4")
        self.assertIsNotNone(move)
        # Illegal move
        move = self.white_agent.is_move_legal(board, "e8")
        self.assertIsNone(move)
        # Invalid notation
        move = self.white_agent.is_move_legal(board, "invalid")
        self.assertIsNone(move)
    
    def test_select_move_opening_book(self):
        """Test that agent uses opening book moves when available"""
        board = chess.Board()
        with patch.object(self.white_agent, 'alpha_beta_search') as mock_search:
            # Should not be called when opening book is used
            mock_search.return_value = (None, 0)
            move = self.white_agent.select_move(board)
            self.assertEqual(board.san(move), "e4")
            self.assertEqual(self.white_agent.opening_moves_played, 1)
            mock_search.assert_not_called()
    
    def test_select_move_magnus(self):
        """Test that black agent uses Magnus moves"""
        board = chess.Board()
        board.push_san("e4")  # white's move
        with patch.object(self.black_agent, 'alpha_beta_search') as mock_search:
            mock_search.return_value = (None, 0)
            move = self.black_agent.select_move(board)
            self.assertEqual(board.san(move), "d4")
            self.assertEqual(self.black_agent.magnus_moves_played, 1)
    
    def test_select_move_alpha_beta(self):
        """Test that agent falls back to alpha-beta search when no book moves available"""
        board = chess.Board()
        white_agent = ChessAgent(chess.WHITE, self.mock_move_book)
        white_agent.opening_moves_played = 999  # Force no opening moves
        white_agent.magnus_moves_played = 999   # Force no Magnus moves
        
        with patch.object(white_agent, 'alpha_beta_search') as mock_search:
            test_move = chess.Move.from_uci("e2e4")
            mock_search.return_value = (test_move, 1.0)
            move = white_agent.select_move(board)
            self.assertEqual(move, test_move)
            mock_search.assert_called()
    
    def test_evaluate(self):
        """Test the evaluation function"""
        board = chess.Board()
        # Default starting position should be roughly equal
        score = self.white_agent.evaluate(board)
        self.assertAlmostEqual(score, 0.0, delta=1.0)
        
        # After e4, white should have a small advantage
        board.push_san("e4")
        score = self.white_agent.evaluate(board)
        self.assertGreater(score, 0)
        
        # Test checkmate evaluation
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 0 1")  # Scholar's mate
        board.push_san("Qf6")
        board.push_san("Qxf7#")
        score = self.white_agent.evaluate(board)
        self.assertGreater(score, 9000)  # Should be close to 10000 (checkmate value)


class TestMatch(unittest.TestCase):
    def setUp(self):
        self.mock_white_agent = MagicMock()
        self.mock_black_agent = MagicMock()
        self.match = Match(self.mock_white_agent, self.mock_black_agent)
    
    def test_initialization(self):
        """Test match initialization"""
        self.assertEqual(self.match.white_agent, self.mock_white_agent)
        self.assertEqual(self.match.black_agent, self.mock_black_agent)
        self.assertEqual(self.match.white_clock, 10.0)
        self.assertEqual(self.match.black_clock, 10.0)
        self.assertTrue(isinstance(self.match.board, chess.Board))
    
    @patch('time.sleep')  # Patch sleep to speed up tests
    @patch('time.time')
    def test_play_one_move(self, mock_time, mock_sleep):
        """Test playing a single move"""
        mock_time.side_effect = [0, 1]  # Simulate 1 second elapsed
        self.mock_white_agent.select_move.return_value = chess.Move.from_uci("e2e4")
        
        with patch.object(self.match, 'render') as mock_render:  # Don't actually render
            with patch.object(self.match.board, 'is_game_over', side_effect=[False, True]):  # End after one move
                result = self.match.play()
                self.mock_white_agent.select_move.assert_called_once()
                self.assertEqual(self.match.white_clock, 9.1)  # 10 - 1 + 0.1 increment
                mock_render.assert_called_once()

    def test_render(self):
        """Test render function (mocked)"""
        with patch('IPython.display.clear_output') as mock_clear:
            with patch('IPython.display.display') as mock_display:
                with patch('chess.svg.board') as mock_svg:
                    self.match.render()
                    mock_clear.assert_called_once()
                    mock_display.assert_called_once()
                    mock_svg.assert_called_once()


if __name__ == '__main__':
    unittest.main()