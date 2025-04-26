import PySimpleGUI as sg
import sys
import threading
import chess
import io
import os
from typing import List, Optional, Tuple

# --- Константы и Настройки ---
SQUARE_SIZE = 60  # Размер клетки стал чуть больше для читаемости
BOARD_SIZE = 8 * SQUARE_SIZE  # Размер самой доски
BORDER_WIDTH = SQUARE_SIZE // 2  # Ширина рамки для букв/цифр
CANVAS_WIDTH = BOARD_SIZE + 2 * BORDER_WIDTH  # Полная ширина холста
CANVAS_HEIGHT = BOARD_SIZE + 2 * BORDER_WIDTH  # Полная высота холста

SCRIPT_DIR = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
PIECE_DIR = os.path.join(SCRIPT_DIR, 'images_chess')
USE_IMAGES = False  # Поставь True для картинок

PIECE_SYMBOLS = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
}
DARK_COLOR = '#B58863'
LIGHT_COLOR = '#F0D9B5'
SELECTED_HIGHLIGHT_COLOR = '#646464'
LEGAL_MOVE_DOT_COLOR = '#3C3C3C'
LEGAL_MOVE_DOT_RADIUS = SQUARE_SIZE // 8

# Цвета для текстовых символов фигур (ИСПРАВЛЕНО)
UNICODE_WHITE_PIECE_COLOR = '#FFFFFF'  # Белый текст для белых фигур
UNICODE_BLACK_PIECE_COLOR = '#000000'  # Черный текст для черных фигур

# Настройки для разметки (НОВОЕ)
LABEL_FONT_SIZE = BORDER_WIDTH // 2
LABEL_COLOR = '#000000'  # Черный цвет для букв/цифр


# --- Перехват print в stderr ---
class StderrRedirector(io.StringIO):
    def write(self, s):
        sys.stderr.write(s)
        sys.stderr.flush()


original_stdout = sys.stdout
sys.stdout = StderrRedirector()


# --- Функции ---

def get_piece_image_data(piece_symbol):
    if not USE_IMAGES: return None
    try:
        color_prefix = 'w' if piece_symbol.isupper() else 'b'
        piece_type = piece_symbol.upper()
        filename = f"{color_prefix}{piece_type}.png"  # Предполагаем PNG
        path = os.path.join(PIECE_DIR, filename)
        if os.path.exists(path):
            if filename.lower().endswith(('.png', '.gif', '.ppm', '.pgm')):
                with open(path, 'rb') as f:
                    return f.read()
            else:  # Попытка конвертации других форматов через Pillow
                print(
                    f"Предупреждение: Формат файла {filename} может не поддерживаться напрямую. Попытка конвертации...",
                    file=sys.stderr)
                try:
                    from PIL import Image
                    img = Image.open(path)
                    with io.BytesIO() as output:
                        img.save(output, format="PNG")
                        return output.getvalue()
                except ImportError:
                    print("Pillow не установлен. Не удалось конвертировать изображение.", file=sys.stderr)
                    return None
                except Exception as e_conv:
                    print(f"Ошибка конвертации {filename}: {e_conv}", file=sys.stderr)
                    return None
        else:
            print(f"Предупреждение: Файл не найден {path}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Ошибка загрузки изображения {piece_symbol}: {e}", file=sys.stderr)
        return None


# --- Функция отрисовки доски (ЗНАЧИТЕЛЬНО ОБНОВЛЕНА) ---
def draw_board(graph: sg.Graph,
               board: chess.Board,
               player_color: chess.Color,
               selected_square: Optional[chess.Square] = None,
               legal_destinations: Optional[List[chess.Square]] = None):
    graph.erase()
    legal_destinations = legal_destinations or []

    # --- 1. Рисуем клетки и фигуры ---
    for vis_row in range(8):  # 0 = верхний визуальный ряд
        for vis_col in range(8):  # 0 = левый визуальный ряд
            # Координаты клетки относительно НИЖНЕГО ЛЕВОГО угла ДОСКИ (не холста)
            # (0,0) доски = нижний левый угол холста Graph
            sq_x0 = vis_col * SQUARE_SIZE
            sq_y0 = (7 - vis_row) * SQUARE_SIZE
            sq_x1 = sq_x0 + SQUARE_SIZE
            sq_y1 = sq_y0 + SQUARE_SIZE

            # Определяем реальную клетку
            if player_color == chess.WHITE:
                file_index = vis_col
                rank_index = 7 - vis_row
            else:
                file_index = 7 - vis_col
                rank_index = vis_row
            sq_index = chess.square(file_index, rank_index)

            # Цвет клетки
            square_bg_color = LIGHT_COLOR if (rank_index + file_index) % 2 == 0 else DARK_COLOR

            # Рисуем фон клетки
            graph.draw_rectangle((sq_x0, sq_y0), (sq_x1, sq_y1), fill_color=square_bg_color, line_color=square_bg_color)

            # Подсветка выбранной
            if selected_square == sq_index:
                graph.draw_rectangle((sq_x0, sq_y0), (sq_x1, sq_y1), fill_color=SELECTED_HIGHLIGHT_COLOR,
                                     line_color=SELECTED_HIGHLIGHT_COLOR)

            # Подсветка легальных ходов
            if sq_index in legal_destinations:
                center_x = sq_x0 + SQUARE_SIZE // 2
                center_y = sq_y0 + SQUARE_SIZE // 2
                graph.draw_circle((center_x, center_y), LEGAL_MOVE_DOT_RADIUS,
                                  fill_color=LEGAL_MOVE_DOT_COLOR, line_color=LEGAL_MOVE_DOT_COLOR)

            # Рисуем фигуру
            piece = board.piece_at(sq_index)
            if piece:
                piece_symbol = piece.symbol()
                # --- ИСПРАВЛЕННЫЙ ЦВЕТ СИМВОЛОВ ---
                piece_text_color = UNICODE_WHITE_PIECE_COLOR if piece.color == chess.WHITE else UNICODE_BLACK_PIECE_COLOR
                # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

                center_x = sq_x0 + SQUARE_SIZE // 2
                center_y = sq_y0 + SQUARE_SIZE // 2
                img_bottom_left_x = sq_x0
                img_bottom_left_y = sq_y0

                if USE_IMAGES:
                    image_data = get_piece_image_data(piece_symbol)
                    if image_data:
                        try:
                            graph.draw_image(data=image_data, location=(img_bottom_left_x, img_bottom_left_y))
                        except Exception as e_img:
                            print(f"Ошибка graph.draw_image для {piece_symbol}: {e_img}", file=sys.stderr)
                            graph.draw_text(PIECE_SYMBOLS.get(piece_symbol, '?'), (center_x, center_y),
                                            font=('Arial', SQUARE_SIZE // 2), color=piece_text_color)  # Fallback
                    else:  # Image not found/error
                        graph.draw_text(PIECE_SYMBOLS.get(piece_symbol, '?'), (center_x, center_y),
                                        font=('Arial', SQUARE_SIZE // 2), color=piece_text_color)  # Fallback
                else:  # Use Unicode symbols
                    graph.draw_text(PIECE_SYMBOLS.get(piece_symbol, '?'), (center_x, center_y),
                                    font=('Arial', SQUARE_SIZE // 2), color=piece_text_color)  # Use corrected color

    # --- 2. Рисуем разметку (буквы и цифры) ---
    label_font = ('Arial', LABEL_FONT_SIZE)
    for i in range(8):
        # Буквы (Файлы)
        file_char = chr(ord('a') + i) if player_color == chess.WHITE else chr(ord('h') - i)
        text_x = i * SQUARE_SIZE + SQUARE_SIZE // 2  # Центр колонки
        text_y_bottom = -BORDER_WIDTH // 2  # Середина нижней рамки
        text_y_top = BOARD_SIZE + BORDER_WIDTH // 2  # Середина верхней рамки
        graph.draw_text(file_char, (text_x, text_y_bottom), color=LABEL_COLOR, font=label_font)
        graph.draw_text(file_char, (text_x, text_y_top), color=LABEL_COLOR, font=label_font)  # Добавим и сверху

        # Цифры (Ранги)
        rank_char = str(8 - i) if player_color == chess.WHITE else str(i + 1)
        text_x_left = -BORDER_WIDTH // 2  # Середина левой рамки
        text_x_right = BOARD_SIZE + BORDER_WIDTH // 2  # Середина правой рамки
        text_y = i * SQUARE_SIZE + SQUARE_SIZE // 2  # Центр ряда (относительно ВЕРХА доски)
        graph.draw_text(rank_char, (text_x_left, text_y), color=LABEL_COLOR, font=label_font)
        graph.draw_text(rank_char, (text_x_right, text_y), color=LABEL_COLOR, font=label_font)  # Добавим и справа


# --- Функция конвертации координат (ОБНОВЛЕНА под рамку) ---
def coord_to_square(x: int, y: int, player_color: chess.Color) -> Optional[chess.Square]:
    """Преобразует координаты клика PySimpleGUI Graph в клетку chess.Square, учитывая ориентацию и рамку."""
    # Проверяем, что клик ВНУТРИ самой доски (0,0) - (BOARD_SIZE, BOARD_SIZE)
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        # print(f"Клик вне доски: ({x}, {y})", file=sys.stderr) # Раскомментируй для отладки кликов
        return None  # Клик был на рамке или вне холста

    # Координаты клика теперь внутри доски, считаем как раньше
    vis_col = x // SQUARE_SIZE
    # (0,0) Graph = нижний левый угол. y растет вверх.
    vis_row = 7 - (y // SQUARE_SIZE)  # 0 = верхний визуальный ряд, 7 = нижний

    # Конвертируем визуальные координаты в шахматные
    if player_color == chess.WHITE:
        file_index = vis_col
        rank_index = 7 - vis_row
    else:
        file_index = 7 - vis_col
        rank_index = vis_row

    if 0 <= file_index < 8 and 0 <= rank_index < 8:
        return chess.square(file_index, rank_index)
    else:
        # Эта ситуация не должна возникать, если проверка выше прошла
        print(f"Warning: Invalid chess indices calculated: file={file_index}, rank={rank_index}", file=sys.stderr)
        return None


# --- Поток чтения Stdin (без изменений) ---
def read_stdin_thread(window: sg.Window):
    print("GUI Stdin Reader: Поток запущен.", file=sys.stderr)
    try:
        for line in sys.__stdin__:
            command = line.strip()
            print(f"GUI Stdin Reader: Получено: '{command}'", file=sys.stderr)
            if command:
                window.write_event_value('-STDIN_COMMAND-', command)
            if command == "stop_game":
                window.write_event_value('-STDIN_COMMAND-', command)
                break
    except EOFError:
        print("GUI Stdin Reader: EOF received, stdin closed.", file=sys.stderr)
        window.write_event_value('-STDIN_ERROR-', "EOF")
    except Exception as e:
        print(f"GUI Stdin Reader: Ошибка: {e}", file=sys.stderr)
        window.write_event_value('-STDIN_ERROR-', str(e))
    finally:
        print("GUI Stdin Reader: Поток завершен.", file=sys.stderr)


# --- Основная функция GUI (ОБНОВЛЕНА под новый размер холста) ---
def main_gui():
    global sys, original_stdout
    print(f"PySimpleGUI Version: {sg.version}", file=sys.stderr)
    sg.theme('SystemDefault')

    # Используем новые константы размера холста
    layout = [
        [sg.Graph(
            canvas_size=(CANVAS_WIDTH, CANVAS_HEIGHT),
            graph_bottom_left=(0, 0),  # НИЖНИЙ ЛЕВЫЙ УГОЛ ДОСКИ внутри холста
            graph_top_right=(BOARD_SIZE, BOARD_SIZE),  # ВЕРХНИЙ ПРАВЫЙ УГОЛ ДОСКИ внутри холста
            key='-BOARD-',
            enable_events=True,
            background_color='lightgrey'  # Фон за доской
        )],
        [sg.Text("Ход: Белых", key='-TURN_STATUS-'), sg.Text("", key='-GAME_STATUS-', size=(30, 1))]
        # Увеличим размер статуса
    ]

    # Добавляем паддинг вокруг Graph, чтобы рамка была видна
    window = sg.Window('PySimpleGUI Шахматы (Stepan)', layout, finalize=True, element_padding=(0, 0))

    graph = window['-BOARD-']
    # --- Перемещаем рамку и рисуем ее один раз ---
    # Мы будем рисовать буквы/цифры в draw_board теперь
    # graph.change_coordinates((0,0), (BOARD_SIZE, BOARD_SIZE)) # Это УЖЕ сделано в Graph

    # --- Остальная инициализация ---
    board = chess.Board()
    selected_square: Optional[chess.Square] = None
    player_color: chess.Color = chess.WHITE
    is_player_turn: bool = False
    legal_destinations: List[chess.Square] = []

    # Первая отрисовка (пустая доска с разметкой)
    draw_board(graph, board, player_color, selected_square, legal_destinations)
    window['-TURN_STATUS-'].update("Ожидание новой игры...")
    window['-GAME_STATUS-'].update("")

    threading.Thread(target=read_stdin_thread, args=(window,), daemon=True).start()

    # Основной цикл событий (логика внутри без изменений)
    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == 'Exit':
            try:
                sys.__stdout__.write("GUI_CLOSED\n")
                sys.__stdout__.flush()
            except Exception as e:
                print(f"GUI Error writing GUI_CLOSED: {e}", file=sys.stderr)
            break

        elif event == '-BOARD-':
            if not is_player_turn: continue
            coords = values['-BOARD-']
            if coords:
                # --- ВАЖНО: Передаем координаты КАК ЕСТЬ в coord_to_square ---
                # Проверка на попадание ВНУТРЬ доски делается внутри coord_to_square
                clicked_square = coord_to_square(coords[0], coords[1], player_color)
                # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

                if clicked_square is not None:  # Клик был на доске, а не на рамке
                    piece = board.piece_at(clicked_square)
                    # (Логика выбора/хода остается такой же, как в предыдущем рабочем варианте)
                    if selected_square is None:
                        if piece and piece.color == player_color:
                            selected_square = clicked_square
                            legal_destinations = [m.to_square for m in board.legal_moves if
                                                  m.from_square == selected_square]
                        else:
                            selected_square = None
                            legal_destinations = []
                    else:
                        if clicked_square == selected_square:
                            selected_square = None
                            legal_destinations = []
                        elif clicked_square in legal_destinations:
                            move = chess.Move(selected_square, clicked_square)
                            from_piece = board.piece_at(selected_square)
                            if from_piece and from_piece.piece_type == chess.PAWN:
                                target_rank = chess.square_rank(clicked_square)
                                if (player_color == chess.WHITE and target_rank == 7) or \
                                        (player_color == chess.BLACK and target_rank == 0):
                                    move.promotion = chess.QUEEN  # Default promotion
                            try:
                                sys.__stdout__.write(f"MOVE:{move.uci()}\n")
                                sys.__stdout__.flush()
                            except Exception as e:
                                print(f"GUI Error writing MOVE: {e}", file=sys.stderr)
                            board.push(move)
                            is_player_turn = False
                            selected_square = None
                            legal_destinations = []
                            window['-TURN_STATUS-'].update(
                                f"Ход: {'Белых' if board.turn == chess.WHITE else 'Черных'} (Ожидание Бота)")
                        elif piece and piece.color == player_color:
                            selected_square = clicked_square
                            legal_destinations = [m.to_square for m in board.legal_moves if
                                                  m.from_square == selected_square]
                        else:
                            selected_square = None
                            legal_destinations = []

                    draw_board(graph, board, player_color, selected_square, legal_destinations)
                    if board.is_checkmate():
                        window['-GAME_STATUS-'].update("Мат!")
                    elif board.is_stalemate():
                        window['-GAME_STATUS-'].update("Пат!")
                    elif board.is_insufficient_material():
                        window['-GAME_STATUS-'].update("Ничья (фигуры)")
                    elif board.is_check():
                        window['-GAME_STATUS-'].update("Шах!")
                    else:
                        window['-GAME_STATUS-'].update("")
                # else: # Клик был на рамке, ничего не делаем
                #     print("Клик по рамке.", file=sys.stderr)

        elif event == '-STDIN_COMMAND-':
            # (Логика обработки команд stdin остается такой же)
            command = values[event]
            if command.startswith("new_game"):
                parts = command.split(" ", 1)
                if len(parts) > 1:
                    color_str = parts[1].lower()
                    player_color = chess.WHITE if color_str == 'white' else chess.BLACK
                    board.reset()
                    selected_square = None
                    legal_destinations = []
                    is_player_turn = (board.turn == player_color)
                    draw_board(graph, board, player_color, selected_square,
                               legal_destinations)  # Перерисовка с новой ориентацией
                    window['-TURN_STATUS-'].update(
                        f"Ход: {'Белых' if board.turn == chess.WHITE else 'Черных'}{' (Ваш)' if is_player_turn else ''}")
                    window['-GAME_STATUS-'].update("")
                else:
                    print("GUI: Ошибка - команда new_game без указания цвета", file=sys.stderr)
            elif command.startswith("move"):
                parts = command.split(" ", 1)
                if len(parts) > 1:
                    uci_move = parts[1]
                    try:
                        move = chess.Move.from_uci(uci_move)
                        if move in board.legal_moves:
                            board.push(move)
                            is_player_turn = True
                            selected_square = None
                            legal_destinations = []
                            draw_board(graph, board, player_color, selected_square,
                                       legal_destinations)  # Перерисовка после хода бота
                            window['-TURN_STATUS-'].update(
                                f"Ход: {'Белых' if board.turn == chess.WHITE else 'Черных'}{' (Ваш)' if is_player_turn else ''}")
                            if board.is_checkmate():
                                window['-GAME_STATUS-'].update("Мат!")
                            elif board.is_stalemate():
                                window['-GAME_STATUS-'].update("Пат!")
                            elif board.is_insufficient_material():
                                window['-GAME_STATUS-'].update("Ничья (фигуры)")
                            elif board.is_check():
                                window['-GAME_STATUS-'].update("Шах!")
                            else:
                                window['-GAME_STATUS-'].update("")
                        else:
                            print(f"GUI: Ошибка - Нелегальный ход от main: {uci_move}", file=sys.stderr)
                    except ValueError:
                        print(f"GUI: Ошибка - Некорректный UCI от main: {uci_move}", file=sys.stderr)
                    except Exception as e_move:
                        print(f"GUI: Ошибка обработки хода {uci_move}: {e_move}", file=sys.stderr)
                else:
                    print("GUI: Ошибка - команда move без указания хода UCI", file=sys.stderr)
            elif command == "stop_game":
                print("GUI: Получена команда stop_game, закрытие...", file=sys.stderr)
                break
        elif event == '-STDIN_ERROR-':
            print(f"GUI: Ошибка в потоке stdin: {values[event]}", file=sys.stderr)
            sg.popup_error(f"Ошибка связи с основным процессом:\n{values[event]}\nGUI будет закрыт.",
                           title="Ошибка IPC")
            break

    window.close()
    print("GUI: Окно закрыто.", file=sys.stderr)


# --- Точка входа для GUI (без изменений) ---
if __name__ == "__main__":
    original_stdout = sys.stdout
    original_stdin = sys.stdin
    original_stderr = sys.stderr
    sys.stdout = StderrRedirector()
    try:
        main_gui()
    except Exception as e_main:
        print(f"Критическая ошибка в main_gui: {e_main}", file=original_stderr)
        import traceback

        traceback.print_exc(file=original_stderr)
    finally:
        sys.stdout = original_stdout
        sys.stdin = original_stdin
        sys.stderr = original_stderr
        # Эта строка может вывестись в stderr, если восстановление прошло успешно
        print("GUI: Стандартные потоки восстановлены.", file=sys.stderr)