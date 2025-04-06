from base_class_player import Player
from base_class_hexboard import HexBoard
import random
import math
import heapq
from copy import deepcopy

class HexAIPlayer(Player):
    def __init__(self, player_id: int, depth=3):
        super().__init__(player_id) # Llamando al contructor de Player y asignando su player_id
        self.depth = depth # Determina cuántos niveles de jugadas (Propia + oponente) va a analizar en profundidad
        self.opponent_id = 2 if player_id == 1 else 1   # Id del oponente

    def play(self, board: HexBoard) -> tuple:
        _, move = self.minimax(board, self.depth, -math.inf, math.inf, True)
        return move

    def minimax(self, board, depth, alpha, beta, maximizing_player): # alpha = Mejor valor (MAX) que la IA puede asegurar hasta ahora     
                                                                     # beta = Mejor valor (MIN) que el oponente puede asegurar
                                                                     # maximizing_player = Booleano que indica si es o no el turno de la IA

        # Caso base: Si ya se llegó a la profundidad esperada o el juego terminó, evalúa con heurística y devuelve el valor
        if depth == 0 or board.check_connection(self.player_id) or board.check_connection(self.opponent_id):
            return self.evaluate(board), None

        best_move = None
        moves = board.get_possible_moves()
        random.shuffle(moves)  # Para evitar siempre la misma jugada

        # Evaluando jugada: Caso turno de IA
        if maximizing_player: # Caso: Turno de la IA (escoger mejor jugada)
            max_eval = -math.inf
            for move in moves: # Ciclo de evaluación de jugadas disponibles
                new_board = deepcopy(board)
                new_board.place_piece(*move, self.player_id)
                eval, _ = self.minimax(new_board, depth - 1, alpha, beta, False) # Se evalúa cuán bueno es el 'futuro' tras esa jugada
                if eval > max_eval: # Procedemos a quedarnos con la mejor evaluación entre la que teníamos y esta
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha: # Si mi jugada actual no supera a la peor jugada del rival, no tiene sentido seguir explorando esa rama 
                    break
            return max_eval, best_move
        # Evaluando jugada: Caso turno de oponente
        else:
            min_eval = math.inf
            for move in moves:
                new_board = deepcopy(board)
                new_board.place_piece(*move, self.opponent_id)
                eval, _ = self.minimax(new_board, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def evaluate(self, board): # Evaluar el estado actual del tablero (para asignar valores a los nodos)
        return (
            1000 * self.a_star(board, self.opponent_id) -   # Penaliza si el oponente está cerca de ganar
            self.a_star(board, self.player_id) +            # Bonifica si nosotros estamos cerca de ganar
            self.centrality_score(board) +                  # Bonifica posiciones cerca del centro, esto da más flexibilidad y mejores opciones de jugada que una posición cerca del borde o de una esquina
            self.connection_potential(board)                
        )

    def neighbors(self, row, col, board):
        directions_even = [(-1, 0), (1, 0), (-1, 1), (1, 1), (0, -1), (0, 1)]   # Direcciones válidas a conectar para filas pares
        directions_odd = [(-1, 0), (1, 0), (-1, -1), (1, -1), (0, -1), (0, 1)]  # Direcciones válidas a conectar para filas impares
        dirs = directions_even if row % 2 == 0 else directions_odd
        # Obteniendo vecinos
        neighbors = []
        for dr, dc in dirs:
            nr, nc = row + dr, col + dc
            if 0 <= nr < board.size and 0 <= nc < board.size:
                neighbors.append((nr, nc))
        return neighbors

    def a_star(self, board, player_id): # Encontrar el camino mínimo de un lado a otro (o de arriba a abajo)
        size = board.size
        frontier = []
        visited = set()

        def is_goal(r, c):  # Verificar si llegamos al final
            if player_id == 1:
                return c == size - 1
            else:
                return r == size - 1

        # Lados de inicio
        for i in range(size):
            if player_id == 1 and board.board[i][0] in [0, player_id]:
                heapq.heappush(frontier, (0, i, 0))
            elif player_id == 2 and board.board[0][i] in [0, player_id]:
                heapq.heappush(frontier, (0, 0, i))

        while frontier:
            cost, r, c = heapq.heappop(frontier)
            if (r, c) in visited:
                continue
            visited.add((r, c))

            if is_goal(r, c):
                return cost

            for nr, nc in self.neighbors(r, c, board):
                if 0 <= nr < size and 0 <= nc < size:
                    cell = board.board[nr][nc]
                    if cell in [0, player_id]:
                        priority = cost + (1 if cell == 0 else 0)
                        heapq.heappush(frontier, (priority, nr, nc))
        return 9999  # No hay camino posible

    def centrality_score(self, board):
        size = board.size
        score = 0
        center = size // 2
        for i in range(size):
            for j in range(size):
                if board.board[i][j] == self.player_id:
                    dist = abs(center - i) + abs(center - j)
                    score += max(0, 10 - dist)
        return score

    def connection_potential(self, board):
        # Heurística inspirada en el DFS de check_connection: mide cuántos nodos propios están conectados en grupo
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

