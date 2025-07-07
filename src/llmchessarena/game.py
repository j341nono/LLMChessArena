import os
import chess
from llama_cpp import Llama

HOME = os.environ.get("HOME", "/home/user")
model_path_llama = os.path.join(HOME, ".LLMChessArena/models/llama-3.2-3b-instruct-q4_k_m.gguf")
model_path_gemma = os.path.join(HOME, ".LLMChessArena/models/gemma-3-4b-it-Q4_K_M.gguf")

# LLMのロード（CPU常駐）
print("Loading LLaMA3...")
llama3 = Llama(model_path=model_path_llama, n_ctx=2048, n_threads=4)
print("Loading Gemma...")
gemma = Llama(model_path=model_path_gemma, n_ctx=2048, n_threads=4)

def get_llm_move(model: Llama, board: chess.Board):
    prompt = f"""You are a chess master playing a game.
Here is the current board in FEN: {board.fen()}
Please provide your next move in UCI format (e.g. e2e4). Just give the move only, no explanation."""
    response = model(prompt, max_tokens=10, stop=["\n"])
    return response["choices"][0]["text"].strip()

def get_user_choice():
    print("Choose your player model:")
    print("1. LLaMA3")
    print("2. Gemma")
    choice = input("Enter 1 or 2: ").strip()
    return "LLaMA3" if choice == "1" else "Gemma"

def play_game():
    user_model_name = get_user_choice()
    bot_model_name = "Gemma" if user_model_name == "LLaMA3" else "LLaMA3"

    print(f"\nYou ({user_model_name}) will play as White.")
    board = chess.Board()

    models = {
        "LLaMA3": llama3,
        "Gemma": gemma
    }

    while not board.is_game_over():
        print("\n" + str(board) + "\n")
        current_model_name = user_model_name if board.turn == chess.WHITE else bot_model_name
        current_model = models[current_model_name]

        print(f"{current_model_name}'s turn ({'White' if board.turn == chess.WHITE else 'Black'})")

        move_str = get_llm_move(current_model, board)
        print(f"{current_model_name} proposes: {move_str}")

        try:
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                board.push(move)
            else:
                print(f"Illegal move by {current_model_name}: {move_str}")
                print(f"{current_model_name} loses by rule violation!")
                break
        except:
            print(f"Invalid move format by {current_model_name}: {move_str}")
            print(f"{current_model_name} loses by format error!")
            break

    print("\nGame Over.")
    print("Final board:")
    print(board)
    result = board.result()
    print("Result:", result)

    if board.is_checkmate():
        winner = user_model_name if board.turn == chess.BLACK else bot_model_name
        print(f"Checkmate! {winner} wins.")
    elif board.is_stalemate():
        print("Stalemate!")
    elif board.is_insufficient_material():
        print("Draw by insufficient material.")
    else:
        print("Draw or rule violation.")

if __name__ == "__main__":
    play_game()
