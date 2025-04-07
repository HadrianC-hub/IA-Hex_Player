from base_class_player import Player
from base_class_hexboard import HexBoard
import random
import math
import heapq
import time

DIRECTIONS_EVEN = [(-1, 0), (1, 0), (-1, 1), (1, 1), (0, -1), (0, 1)]
DIRECTIONS_ODD = [(-1, 0), (1, 0), (-1, -1), (1, -1), (0, -1), (0, 1)]
TIME_LIMIT = 3.0

class HexAIPlayer(Player):
    def __init__(self, player_id: int):
        super().__init__(player_id) # Llamando al contructor de Player y asignando su player_id
        self.opponent_id = 2 if player_id == 1 else 1   # Id del oponente

    def play(self, board: HexBoard) -> tuple:
        #Iniciar el temporizador
        start_time = time.time()
        self.start_time = start_time

        # Determinar profundidad dinámica (cuántos niveles de jugadas se van a analizar en minimax)
        pos_moves = board.get_possible_moves()
        dynamic_depth = self.get_dynamic_depth(board, len(pos_moves))

        # Usar minimax
        _, move = self.minimax(board, dynamic_depth, -math.inf, math.inf, True)

        # Calcular tiempo y retornar
        duration = time.time() - start_time
        print(f"Tiempo total de jugada: {duration:.3f}s")
        return move

    def get_dynamic_depth(self, board: HexBoard, empty_cells) -> int: # Función para obtener la fase del juego (útil para la profundidad variable)
        total_cells = board.size * board.size
        ratio = empty_cells / total_cells
        if ratio > 0.8:
            return 3
        elif ratio > 0.3:
            return 5
        else:
            return 7

    def minimax(self, board, depth, alpha, beta, maximizing_player): # alpha = Mejor valor (MAX) que la IA puede asegurar hasta ahora     
                                                                     # beta = Mejor valor (MIN) que el oponente puede asegurar
                                                                     # maximizing_player = Booleano que indica si es o no el turno de la IA

        # Cortamos la evaluación si nos pasamos de tiempo
        if time.time() - self.start_time > TIME_LIMIT:
            return self.evaluate(board), None

        # Caso base: Si ya se llegó a la profundidad esperada o el juego terminó, evalúa con heurística y devuelve el valor
        if board.check_connection(self.player_id):
            return math.inf, None
        elif board.check_connection(self.opponent_id):
            return -math.inf, None
        elif depth == 0 or not board.get_possible_moves():
            return self.evaluate(board), None

        best_move = None
        moves = board.get_possible_moves()

        # Ordenar movimientos antes de evaluarlos (se ordenan según la heurística utilizada)
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
        
    def evaluate(self, board: HexBoard) -> float:
        size = board.size
        score = 0
        empty_cells = []

        # Empieza buscando cada celda del jugador y del enemigo, sumando y restando puntos
        for row in range(size):
            for col in range(size):
                cell = board.board[row][col]
                if cell == self.player_id:
                    score += 10
                elif cell == self.opponent_id:
                    score -= 10
                else:
                    empty_cells.append((row, col))  # Guarda las casillas vacías para análisis posterior

        # A* heurístico
        my_path_cost = self.a_star(board, self.player_id)
        opp_path_cost = self.a_star(board, self.opponent_id)
        if opp_path_cost <= 3:
            score -= 300  # Penalización fuerte si el oponente casi conecta
        score += 1000 / (1 + my_path_cost)  # Mientras más costoso sea el camino, menor es la puntuación
        score -= 1000 / (1 + opp_path_cost) # Lo mismo para el análisis del jugador contrario

        # Centralidad y vecinos
        mid = size // 2
        for (r, c) in empty_cells:
            # Cada casilla vacía cerca del centro aumenta el valor (posibles movimientos futuros)
            dist_center = abs(r - mid) + abs(c - mid)
            centrality_bonus = max(0, (size - dist_center))
            score += centrality_bonus * 0.5
            # Aumenta o disminuye el peso de una casilla vacía en dependencia de sus vecinos
            friendly = 0
            enemy = 0
            for nr, nc in self.neighbors(r, c, board):
                neighbor = board.board[nr][nc]
                if neighbor == self.player_id:
                    friendly += 1
                elif neighbor == self.opponent_id:
                    enemy += 1
            score += friendly * 1.5
            score -= enemy * 1.5

        # Bloqueo estratégico del oponente
        for row in range(size):
            for col in range(size):
                cell = board.board[row][col]
                if cell == self.player_id:
                    # Detectar si esta ficha bloquea una dirección "natural" del oponente
                    # Jugador 1: izquierda a derecha → bloqueamos en columnas
                    # Jugador 2: arriba a abajo → bloqueamos en filas

                    if self.opponent_id == 1:
                        importance = col/size
                        # Buscamos si esta ficha está rodeada por fichas enemigas horizontalmente
                        if col > 0 and col < size - 1:
                            if (board.board[row][col - 1] == self.opponent_id and
                                board.board[row][col + 1] == self.opponent_id):
                                score += 20 + (importance*10) # Ajustable según la importancia del bloqueo
                    else:
                        importance = row/size
                        # Buscamos si está rodeada verticalmente
                        if row > 0 and row < size - 1:
                            if (board.board[row - 1][col] == self.opponent_id and
                                board.board[row + 1][col] == self.opponent_id):
                                score += 20 + (importance*10)

        return score

    def a_star(self, board: HexBoard, player_id): # Devuelve el costo mínimo de unir dos lados
        size = board.size
        visited = set()
        heap = []

        def heuristic(row, col): # Heurística: distancia Manhattan al borde opuesto
            return size - 1 - (col if player_id == 1 else row)

        def is_goal(row, col): # Verifica conexiones existentes entre los lados del jugador
            return (col == size - 1) if player_id == 1 else (row == size - 1)

        # Inicialmente se guardan las casillas que tocan uno de los lados en el espacio de búsqueda
        for i in range(size):
            row, col = (i, 0) if player_id == 1 else (0, i)
            if board.board[row][col] == player_id:
                heapq.heappush(heap, (heuristic(row, col), 0, row, col))
            elif board.board[row][col] == 0:
                heapq.heappush(heap, (1 + heuristic(row, col), 1, row, col))

        while heap:
            priority, cost, row, col = heapq.heappop(heap)
            if (row, col) in visited:
                continue
            visited.add((row, col))

            if is_goal(row, col):
                return cost

            for nr, nc in self.neighbors(row, col, board):
                if (nr, nc) in visited:
                    continue
                cell = board.board[nr][nc]
                if cell == player_id:
                    new_cost = cost  # Paso gratis por nuestras fichas
                elif cell == 0:
                    new_cost = cost + 1  # Paso válido, leve coste
                else:
                    continue  # Casilla del oponente, no transitable
                heapq.heappush(heap, (new_cost + heuristic(nr, nc), new_cost, nr, nc))

        return math.inf  # No hay camino posible

