from base_class_player import Player
from base_class_hexboard import HexBoard
import random
import math
import heapq

DIRECTIONS_EVEN = [(-1, 0), (1, 0), (-1, 1), (1, 1), (0, -1), (0, 1)]
DIRECTIONS_ODD = [(-1, 0), (1, 0), (-1, -1), (1, -1), (0, -1), (0, 1)]

class HexAIPlayer(Player):
    def __init__(self, player_id: int, depth=3):
        super().__init__(player_id) # Llamando al contructor de Player y asignando su player_id
        self.depth = depth # Determina cuántos niveles de jugadas (Propia + oponente) va a analizar en profundidad
        self.opponent_id = 2 if player_id == 1 else 1   # Id del oponente

    def play(self, board: HexBoard) -> tuple:
        # Verificar si hay una jugada que gana inmediatamente
        pos_moves = board.get_possible_moves()
        length_moves = len(pos_moves)
        for move in pos_moves:
            temp_board = board.clone()
            temp_board.place_piece(*move, self.player_id)
            if temp_board.check_connection(self.player_id):
                return move

        # Si no hay jugada ganadora inmediata, usar minimax
        dynamic_depth = self.get_dynamic_depth(board, length_moves)
        _, move = self.minimax(board, dynamic_depth, -math.inf, math.inf, True)
        return move

    def get_dynamic_depth(self, board: HexBoard, empty_cells) -> int: # Función para obtener la fase del juego (útil para la profundidad variable)
        
        total_cells = board.size * board.size
        ratio = empty_cells / total_cells

        if ratio > 0.75:
            return 2  # Early game
        elif ratio > 0.5:
            return 3  # Mid-early
        elif ratio > 0.25:
            return 4  # Mid-late
        else:
            return 5  # Endgame

    def minimax(self, board, depth, alpha, beta, maximizing_player): # alpha = Mejor valor (MAX) que la IA puede asegurar hasta ahora     
                                                                     # beta = Mejor valor (MIN) que el oponente puede asegurar
                                                                     # maximizing_player = Booleano que indica si es o no el turno de la IA

        # Caso base: Si ya se llegó a la profundidad esperada o el juego terminó, evalúa con heurística y devuelve el valor
        if board.check_connection(self.player_id):
            return math.inf, None
        elif board.check_connection(self.opponent_id):
            return -math.inf, None
        elif depth == 0 or not board.get_possible_moves():
            return self.evaluate(board), None

        best_move = None
        moves = board.get_possible_moves()

        # Ordenar movimientos antes de evaluarlos según la heurística utilizada
        if maximizing_player: # Si estamos maximizando al jugador, ordenar jugadas por valores de mayor a menor según la heurística
            moves.sort(key=lambda move: self.evaluate_after_move(board, move, self.player_id), reverse=True)
        else:   # Si estamos minimizando, ordenar jugadas por valores de menor a mayor segun la heurística
            moves.sort(key=lambda move: self.evaluate_after_move(board, move, self.opponent_id), reverse=False)

        # Evaluando jugada: Caso turno de IA
        if maximizing_player: # Caso: Turno de la IA (escoger mejor jugada)
            max_eval = -math.inf
            for move in moves: # Ciclo de evaluación de jugadas disponibles
                new_board = board.clone()
                new_board.place_piece(*move, self.player_id)
                eval, _ = self.minimax(new_board, depth - 1, alpha, beta, False) # Se evalúa cuán bueno es el 'futuro' tras esa jugada
                if eval > max_eval: # Procedemos a quedarnos con la mejor evaluación entre la que teníamos y esta
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha: # Si mi jugada actual no supera a la peor jugada del rival, no tiene sentido seguir explorando esa rama 
                    break
            if best_move is None and moves:
                # Si no se eligió jugada por poda u otra razón, elegimos una jugada para molestar al rival
                fallback_move = self.defensive_fallback_move(board, moves)
                return max_eval, fallback_move
            return max_eval, best_move
        # Evaluando jugada: Caso turno de oponente
        else:
            min_eval = math.inf
            for move in moves:
                new_board = board.clone()
                new_board.place_piece(*move, self.opponent_id)
                eval, _ = self.minimax(new_board, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            if best_move is None and moves:
                # Si no se eligió jugada por poda u otra razón, elegimos una jugada para molestar al rival
                fallback_move = self.defensive_fallback_move(board, moves)
                return min_eval, fallback_move
            return min_eval, best_move

    def evaluate_after_move(self, board: HexBoard, move, player_id): # Evalúa rápidamente un tablero como si el jugador hiciera esa jugada
        temp_board = board.clone()
        temp_board.place_piece(*move, player_id)
        return self.evaluate(temp_board)

    def evaluate(self, board):
        my_weights = self.a_star_weighted_moves(board, self.player_id)
        opponent_weights = self.a_star_weighted_moves(board, self.opponent_id)

        my_score = sum(my_weights.values())
        opponent_score = sum(opponent_weights.values())
        
        connection_score = self.connection_potential(board)
        blocking_score = self.blocking_potential(board)  # Añadimos el score por interrumpir cadenas

        # Evaluación final: favorece conexiones propias, interrumpir al oponente, y da un poco más de peso al centro en fases iniciales
        return my_score - opponent_score + connection_score - blocking_score

    def neighbors(self, row, col, board):
        dirs = DIRECTIONS_EVEN if row % 2 == 0 else DIRECTIONS_ODD
        # Obteniendo vecinos
        neighbors = []
        size = board.size
        for dr, dc in dirs:
            nr, nc = row + dr, col + dc
            if 0 <= nr < size and 0 <= nc < size:
                neighbors.append((nr, nc))
        return neighbors

    def a_star_weighted_moves(self, board, player_id):
        size = board.size
        frontier = []
        visited = set()
        weight_map = {}

        def heuristic(r, c):
            return size - 1 - c if player_id == 1 else size - 1 - r

        def is_goal(r, c):
            return c == size - 1 if player_id == 1 else r == size - 1

        for i in range(size):
            if player_id == 1 and board.board[i][0] in [0, player_id]:
                heapq.heappush(frontier, (0 + heuristic(i, 0), 0, i, 0))
            elif player_id == 2 and board.board[0][i] in [0, player_id]:
                heapq.heappush(frontier, (0 + heuristic(0, i), 0, 0, i))

        while frontier:
            priority, cost, r, c = heapq.heappop(frontier)
            if (r, c) in visited:
                continue
            visited.add((r, c))

            if board.board[r][c] == 0:
                bonus = self.centrality_bonus(size, r, c) + self.connection_bonus(board, r, c, player_id)
                weight_map[(r, c)] = weight_map.get((r, c), 0) + bonus

            if is_goal(r, c):
                continue

            for nr, nc in self.neighbors(r, c, board):
                if 0 <= nr < size and 0 <= nc < size:
                    cell = board.board[nr][nc]
                    if cell in [0, player_id]:
                        new_cost = cost + (1 if cell == 0 else 0)
                        heapq.heappush(frontier, (new_cost + heuristic(nr, nc), new_cost, nr, nc))

        return weight_map

    def centrality_bonus(self, size, r, c): # Agrega un bono a las casillas cercanas al centro (Usada en A*)
        center = size // 2
        dist = abs(center - r) + abs(center - c)
        return max(0, 10 - dist)

    def connection_bonus(self, board, r, c, player_id): # Agrega un bono a las casillas cercanas de otras casillas del jugador (Usada en A*)
        nearby = 0
        for nr, nc in self.neighbors(r, c, board):
            if board.board[nr][nc] == player_id:
                nearby += 1
        return nearby * 10

    def connection_potential(self, board): # Heurística inspirada en el DFS de check_connection: mide cuántos nodos propios están conectados en grupo
        size = board.size
        visited = set()
        score = 0

        def dfs(r, c):
            stack = [(r, c)]
            connected = 0
            while stack:
                r, c = stack.pop()
                if (r, c) in visited:
                    continue
                visited.add((r, c))
                connected += 1
                for nr, nc in self.neighbors(r, c, board):
                    if 0 <= nr < size and 0 <= nc < size:
                        if board.board[nr][nc] == self.player_id and (nr, nc) not in visited:
                            stack.append((nr, nc))
            return connected

        for i in range(size):
            for j in range(size):
                if board.board[i][j] == self.player_id and (i, j) not in visited:
                    connected = dfs(i, j)
                    score += connected ** 2  # Más peso a grupos grandes
        return score

    def blocking_potential(self, board):
        size = board.size
        visited = set()
        score = 0

        def dfs(r, c):
            stack = [(r, c)]
            connected = 0
            while stack:
                r, c = stack.pop()
                if (r, c) in visited:
                    continue
                visited.add((r, c))
                connected += 1
                for nr, nc in self.neighbors(r, c, board):
                    if 0 <= nr < size and 0 <= nc < size:
                        if board.board[nr][nc] == self.opponent_id and (nr, nc) not in visited:
                            stack.append((nr, nc))
            return connected

        for i in range(size):
            for j in range(size):
                if board.board[i][j] == self.opponent_id and (i, j) not in visited:
                    connected = dfs(i, j)
                    score += connected ** 2
        return score

    def defensive_fallback_move(self, board, moves):
        # Evalúa qué jugada complica más al oponente (minimiza su evaluación)
        min_opponent_eval = math.inf
        best_defensive_move = None

        for move in moves:
            eval_for_opponent = self.evaluate_after_move(board, move, self.opponent_id)
            if eval_for_opponent < min_opponent_eval:
                min_opponent_eval = eval_for_opponent
                best_defensive_move = move

        # Si no encuentra jugada útil, elige una aleatoria como última opción
        if best_defensive_move is None and moves:
            best_defensive_move = random.choice(moves)

        return best_defensive_move