import os
import sys
import chess
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# --- 定数定義 ---
HOME = os.environ.get("HOME", "/home/user")
MODEL_DIR = os.path.join(HOME, ".LLMChessArena/models")

# Hugging Face上のモデル情報
MODELS_INFO = {
    "llama3": {
        "repo_id": "unsloth/llama-3.2-3b-instruct-GGUF",
        "filename": "llama-3.2-3b-instruct-q4_k_m.gguf",
    },
    "gemma": {
        "repo_id": "unsloth/gemma-3-4b-it-GGUF",
        "filename": "gemma-3-4b-it-Q4_K_M.gguf",
    }
}

# --- グローバル変数 ---
llama3 = None
gemma = None

def prepare_models():
    """
    モデルファイルが存在しない場合にHugging Face Hubからダウンロードし、
    モデルへのパスを返す。
    """
    global llama3, gemma
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_paths = {}
    for name, info in MODELS_INFO.items():
        model_path = os.path.join(MODEL_DIR, info["filename"])
        if not os.path.exists(model_path):
            print(f"Downloading {name} model: {info['filename']}...")
            try:
                hf_hub_download(
                    repo_id=info["repo_id"],
                    filename=info["filename"],
                    local_dir=MODEL_DIR,
                    local_dir_use_symlinks=False
                )
                print("Download complete.")
            except Exception as e:
                print(f"Error downloading {name} model: {e}", file=sys.stderr)
                sys.exit(1)
        model_paths[name] = model_path

    # LLMのロード（CPU常駐）
    # verbose=False を指定して詳細ログを抑制
    print("Loading LLaMA3 model into memory...")
    llama3 = Llama(model_path=model_paths["llama3"], n_ctx=2048, n_threads=os.cpu_count(), verbose=False)
    print("Loading Gemma model into memory...")
    gemma = Llama(model_path=model_paths["gemma"], n_ctx=2048, n_threads=os.cpu_count(), verbose=False)
    print("Models loaded and ready. 🎮")


def get_llm_move(model: Llama, board: chess.Board):
    """LLMに次の手を生成させる"""
    # FEN形式で現在の盤面を伝え、UCI形式での応答を要求するプロンプト
    prompt = f"""You are a chess master. The current board state is given in FEN notation.
FEN: {board.fen()}
Your task is to provide the next move in UCI format (e.g., e2e4).
Provide only the move, with no explanation or any other text.
Move:"""

    try:
        response = model(prompt, max_tokens=10, stop=["\n", " "])
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(f"Error getting move from LLM: {e}", file=sys.stderr)
        return "error"


def get_player_choice():
    """ユーザーが操作するモデルを選択させる"""
    while True:
        print("\n🤖 Choose your chess engine:")
        print("1. LLaMA3")
        print("2. Gemma")
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return "LLaMA3", "Gemma"
        elif choice == "2":
            return "Gemma", "LLaMA3"
        else:
            print("Invalid input. Please enter 1 or 2.")


def main():
    """ゲームのメインロジック"""
    # 1. モデルの準備とロード
    prepare_models()

    # 2. プレイヤーの選択
    user_model_name, bot_model_name = get_player_choice()
    print(f"\nGame Start! You are playing as {user_model_name} (White).")
    print(f"The opponent is {bot_model_name} (Black).\n")

    board = chess.Board()
    models = {"LLaMA3": llama3, "Gemma": gemma}

    # 3. ゲームループ
    while not board.is_game_over(claim_draw=True):
        print("----------------------------------------")
        print(board)
        print("----------------------------------------")

        is_white_turn = board.turn == chess.WHITE
        current_model_name = user_model_name if is_white_turn else bot_model_name
        current_model = models[current_model_name]
        turn_color = "White" if is_white_turn else "Black"

        print(f"Turn {board.fullmove_number}: {current_model_name}'s move as {turn_color}")

        move_uci = get_llm_move(current_model, board)
        print(f"{current_model_name} proposes move: {move_uci}")

        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
            else:
                print(f"⚠️ Illegal move by {current_model_name}: {move_uci}")
                winner = bot_model_name if is_white_turn else user_model_name
                print(f"'{current_model_name}' loses by illegal move! Winner: {winner}")
                break
        except ValueError:
            print(f"❌ Invalid move format by {current_model_name}: {move_uci}")
            winner = bot_model_name if is_white_turn else user_model_name
            print(f"'{current_model_name}' loses by invalid format! Winner: {winner}")
            break

    # 4. 結果表示
    print("\n========== GAME OVER ==========")
    print("Final board state:")
    print(board)
    print(f"Result: {board.result(claim_draw=True)}")

    if board.is_checkmate():
        winner_color = "Black" if board.turn == chess.WHITE else "White"
        winner_model = bot_model_name if winner_color == "Black" else user_model_name
        print(f"🏆 Checkmate! {winner_model} ({winner_color}) wins!")
    elif board.is_stalemate():
        print("🤝 Stalemate! It's a draw.")
    elif board.is_insufficient_material():
        print("🤝 Draw due to insufficient material.")
    elif board.is_seventyfive_moves():
        print("🤝 Draw by the 75-move rule.")
    elif board.is_fivefold_repetition():
        print("🤝 Draw by fivefold repetition.")
    elif board.is_variant_draw():
        print("🤝 Draw by game-specific rules.")


if __name__ == "__main__":
    main()