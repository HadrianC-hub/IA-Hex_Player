class HexBoard:
    """ En esta clase se implementa el tablero y las funciones necesarias para su manejo"""
    def __init__(self, size: int):
        self.size = size  # Tamaño N del tablero (NxN)
        self.board = [[0 for _ in range(size)] for _ in range(size)]  # Matriz NxN (0=vacío, 1=Jugador1, 2=Jugador2)
         
    def clone(self) -> "HexBoard":
        """Devuelve una copia del tablero actual"""
        board = HexBoard(self.size)
        board.board = [row[:] for row in self.board]  # Hacer una copia profunda de cada fila
        return board

    def place_piece(self, row: int, col: int, player_id: int) -> bool:
        """Coloca una ficha si la casilla está vacía"""
        if self.board[row][col] == 0:
            self.board[row][col] = player_id
            return True # Caso en que una casilla es colocada
        return False    # Caso en que una casilla no se pudo colocar
        
    def get_possible_moves(self) -> list:
        """Devuelve todas las casillas vacías como tuplas (fila, columna)"""
        result = []
        for i in range(self.size):
            for j in range (self.size):
                if self.board[i][j] == 0:
                    result.append((i,j))
        return result
   
    def check_connection(self, player_id: int) -> bool:
        """Verifica si el jugador ha conectado sus dos lados"""

        visited = [[False for _ in range(self.size)] for _ in range(self.size)] # Al inicio, ningún nodo ha sido visitado
        stack = []                                                              # Stack vacío

        # Definiendo función de adyacencia para obtener vecinos
        def get_neighbors(row, col):
            directions = [(-1, 0), (1, 0), (-1, 1), (1, -1), (0, -1), (0, 1)]   # Direcciones válidas a conectar
            dirs = directions

            # Obteniendo vecinos
            neighbors = []
            for dr, dc in dirs:
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    neighbors.append((nr, nc))
            return neighbors

        # Empezar desde el borde correspondiente al jugador
        if player_id == 1:
            # Conectar izquierda a derecha
            for row in range(self.size):
                if self.board[row][0] == 1:
                    stack.append((row, 0))
                    visited[row][0] = True
        else:
            # Conectar arriba a abajo
            for col in range(self.size):
                if self.board[0][col] == 2:
                    stack.append((0, col))
                    visited[0][col] = True

        # DFS para obtener un camino desde un borde al opuesto (usando los valores en el stack)
        while stack:
            r, c = stack.pop()
            if player_id == 1 and c == self.size - 1:
                return True  # Alcanzó el borde derecho
            if player_id == 2 and r == self.size - 1:
                return True  # Alcanzó el borde inferior

            neighbors = get_neighbors(r, c)
            for nr, nc in neighbors:
                if not visited[nr][nc] and self.board[nr][nc] == player_id:
                    visited[nr][nc] = True
                    stack.append((nr, nc))

        return False  # No se encontró conexión
