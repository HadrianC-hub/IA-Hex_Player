import sys
import time
from base_class_hexboard import HexBoard
from base_class_player import Player
from player import HexAIPlayer

# Colores ANSI
RED = "1"
BLUE = "2"
EMPTY = "."

class HumanPlayer(Player):
    def play(self, board: HexBoard) -> tuple:
        while True:
            try:
                move = input(f"Jugador {self.player_id} - Ingresa tu movimiento (fila,columna): ")
                row, col = map(int, move.strip().split(","))
                if (row, col) in board.get_possible_moves():
                    return (row, col)
                else:
                    print("Movimiento inválido. Intenta de nuevo.")
            except Exception:
                print("Formato incorrecto. Usa: fila,columna (ej. 1,2)")

def print_board(board: HexBoard):
    print("\nTablero actual:")
    for i in range(board.size):
        print(" " * i, end="")  # Indentación para simular tablero hexagonal
        for j in range(board.size):
            cell = board.board[i][j]
            if cell == 0:
                print(EMPTY, end=" ")
            elif cell == 1:
                print(RED, end=" ")
            elif cell == 2:
                print(BLUE, end=" ")
        print()
    print()

def choose_players():
    print("Elige modo de juego:")
    print("1. Jugador vs Jugador")
    print("2. Jugador vs IA")
    print("3. IA vs IA")
    choice = input("Opción (1/2/3): ")

    if choice == "1":
        return HumanPlayer(1), HumanPlayer(2)
    elif choice == "2":
        return HumanPlayer(1), HexAIPlayer(2)
    elif choice == "3":
        return HexAIPlayer(1), HexAIPlayer(2)
    else:
        print("Opción inválida.")
        sys.exit(1)

def main():
    try:
        size = int(input("Tamaño del tablero (recomendado 5-7): "))
        if size < 2:
            print("El tamaño mínimo del tablero es 2.")
            return
    except ValueError:
        print("Debes ingresar un número entero.")
        return

    board = HexBoard(size)
    player1, player2 = choose_players()
    current_player = player1

    print_board(board)

    while True:
        move = current_player.play(board)
        success = board.place_piece(move[0], move[1], current_player.player_id)

        if not success:
            print("Movimiento inválido, esa casilla ya está ocupada.")
            continue

        print_board(board)

        if board.check_connection(current_player.player_id):
            print(f"Jugador {current_player.player_id} ha ganado el juego.")
            break

        current_player = player2 if current_player == player1 else player1

        if isinstance(current_player, HexAIPlayer):
            time.sleep(0.5)

if __name__ == "__main__":
    main()