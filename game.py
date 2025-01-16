#code that was ouputted by AI from the first test of echo

def display_board(board):
    """Displays the current state of the board."""
    print(f"\n {board[0]} | {board[1]} | {board[2]} ")
    print("---|---|---")
    print(f" {board[3]} | {board[4]} | {board[5]} ")
    print("---|---|---")
    print(f" {board[6]} | {board[7]} | {board[8]} \n")


def check_winner(board, player):
    """Checks if the player has won the game."""
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    for condition in win_conditions:
        if all(board[i] == player for i in condition):
            return True
    return False


def get_player_move(board):
    """Prompts the player to enter a move."""
    while True:
        try:
            move = int(input("Enter your move (1-9): ")) - 1
            if move in range(9) and board[move] == ' ':
                return move
            print("Invalid move. Try again.")
        except ValueError:
            print("Invalid input. Enter a number between 1 and 9.")


def bot_move(board):
    """Determines the bot's move using a basic strategy."""
    # Play center if available
    if board[4] == ' ':
        return 4

    # Check for possible winning moves or blocks
    for player in ['O', 'X']:
        for i in range(9):
            if board[i] == ' ':
                board[i] = player
                if check_winner(board, player):
                    board[i] = ' '
                    return i
                board[i] = ' '

    # Take any available corner
    for i in [0, 2, 6, 8]:
        if board[i] == ' ':
            return i

    # Take any available side
    for i in [1, 3, 5, 7]:
        if board[i] == ' ':
            return i


def main():
    board = [' '] * 9  # Initialize empty board
    current_player = 'X'  # Player starts as 'X'

    while True:
        display_board(board)
        if current_player == 'X':
            move = get_player_move(board)
        else:
            move = bot_move(board)

        board[move] = current_player
        if check_winner(board, current_player):
            display_board(board)
            print(f"Player {current_player} wins!")
            break
        elif ' ' not in board:
            display_board(board)
            print("It's a draw!")
            break

        current_player = 'O' if current_player == 'X' else 'X'


if __name__ == "__main__":
    main()
