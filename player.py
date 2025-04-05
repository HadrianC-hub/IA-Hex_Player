from base_class_player import Player
from base_class_hexboard import HexBoard
import random

class HexAIPlayer(Player):
    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.opponent_id = 2 if player_id == 1 else 1

    def play(self, board: HexBoard) -> tuple:
        possible_moves = board.get_possible_moves()
        # TODO: Aquí irá la lógica inteligente. Por ahora, jugada aleatoria.
        return random.choice(possible_moves)
