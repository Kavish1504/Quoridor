import pygame
import sys
import copy
import time
WIDTH, HEIGHT = 720, 720
ROWS, COLS = 9, 9
CELL_SIZE = WIDTH // COLS
WALL_THICKNESS = 10
FPS = 60

BLUE = (66, 135, 245)
RED = (245, 66, 66)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (139, 69, 19)
GREY = (211, 211, 211)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Quoridor - Minimax with Alpha-Beta Pruning")
clock = pygame.time.Clock()

class Board:
    def __init__(self):
        self.size = 9
        # self.p1_pos=(0,0)
        # self.p1_pos=(8,0)
        self.p1_pos = (0, 4)
        # self.p2_pos = (8, 0)
        # self.p2_pos = (6, 0)
        self.p2_pos = (8, 4)
        self.p1_walls = 10
        self.p2_walls = 10
        self.h_walls = set()
        self.v_walls = set()

    def is_wall_between(self, pos1, pos2):
        row1, col1 = pos1
        row2, col2 = pos2
        if abs(row1 - row2) + abs(col1 - col2) != 1:
            return False
    
        if col1 == col2:
            min_row = min(row1, row2)
            if (min_row, col1) in self.h_walls or (min_row, col1-1) in self.h_walls:
                return True
        elif row1 == row2:
            min_col = min(col1, col2)
            if (row1, min_col) in self.v_walls or (row1-1, min_col) in self.v_walls:
                return True
                
        return False

    def get_neighbors(self, pos):
        row, col = pos
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                if not self.is_wall_between(pos, (new_row, new_col)):
                    neighbors.append((new_row, new_col))
                    
        return neighbors

    def move_player(self, pos, is_player1):
        if is_player1:
            self.p1_pos = pos
        else:
            self.p2_pos = pos

    def game_over(self):
        return self.p1_pos[0] == 8 or self.p2_pos[0] == 0

    def get_winner(self):
        if self.p1_pos[0] == 8:
            return 1 
        elif self.p2_pos[0] == 0:
            return 2  
        return 0 

    def path_exists(self, start, goal_row):
        visited = set()
        queue = [start]
        
        while queue:
            current = queue.pop(0)
            row, col = current
            if row == goal_row:
                return True
                
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
        return False

    def path_cost(self, start, goal_row):
        visited = set()
        queue = [(start, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            row, col = current
            if row == goal_row:
                return dist
                
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
                    
        return 999

    def clone(self):
        return copy.deepcopy(self)

    def get_possible_moves(self, pos):
        return self.get_neighbors(pos)

    def place_wall(self, orientation, row, col, is_player1):
        if (is_player1 and self.p1_walls <= 0) or (not is_player1 and self.p2_walls <= 0):
            return False
            
        if not self.valid_wall(orientation, row, col):
            return False
            
        if orientation == 'h':
            self.h_walls.add((row, col))
        else:
            self.v_walls.add((row, col))
            
        if is_player1:
            self.p1_walls -= 1
        else:
            self.p2_walls -= 1
            
        return True

    def valid_wall(self, orientation, row, col):
        if row < 0 or col < 0 or row >= self.size - 1 or col >= self.size - 1:
            return False

        if orientation == 'h':
            if (row, col) in self.h_walls or (row, col+1) in self.h_walls:
                return False
           
            for c in range(col, col+2):
                if (row, c) in self.v_walls and (row+1, c) in self.v_walls:
                    return False
        else:
            if (row, col) in self.v_walls or (row+1, col) in self.v_walls:
                return False
            
            for r in range(row, row+2):
                if (r, col) in self.h_walls and (r, col+1) in self.h_walls:
                    return False

        temp_board = self.clone()
        if orientation == 'h':
            temp_board.h_walls.add((row, col))
        else:
            temp_board.v_walls.add((row, col))

        if not temp_board.path_exists(temp_board.p1_pos, 8) or not temp_board.path_exists(temp_board.p2_pos, 0):
            return False

        return True

    def get_valid_wall_placements(self, is_player1):
        walls = []
        if (is_player1 and self.p1_walls > 0) or (not is_player1 and self.p2_walls > 0):
            for orientation in ['h', 'v']:
                for row in range(8):
                    for col in range(8):
                        if self.valid_wall(orientation, row, col):
                            walls.append((orientation, row, col))
        return walls

    def get_all_possible_actions(self, is_player1):
        actions = []
        pos = self.p1_pos if is_player1 else self.p2_pos
        for move in self.get_possible_moves(pos):
            actions.append(('move', move))
    
        for wall in self.get_valid_wall_placements(is_player1):
            actions.append(('wall', wall))
            
        return actions

    def apply_action(self, action, is_player1):
        action_type, action_data = action
        if action_type == 'move':
            self.move_player(action_data, is_player1)
            return True
        elif action_type == 'wall':
            orientation, row, col = action_data
            return self.place_wall(orientation, row, col, is_player1)
        
        return False

    def evaluate(self):
        # Heuristic function for board evaluation.
        # Positive values favor player 2 (AI), negative values favor player 1
        p1_distance = self.path_cost(self.p1_pos, 8)
        p2_distance = self.path_cost(self.p2_pos, 0)
        
        wall_advantage = self.p2_walls - self.p1_walls
        
        return p1_distance - p2_distance + wall_advantage * 0.5

    def debug_walls(self):
        print("Horizontal walls:", self.h_walls)
        print("Vertical walls:", self.v_walls)

def minimax(board, depth, alpha, beta, is_maximizing, max_depth=3):
    # Minimax algorithm with Alpha-Beta pruning
    # - board: current board state
    # - depth: current depth in the search tree
    # - alpha, beta: alpha-beta pruning parameters
    # - is_maximizing: True if maximizing player's turn (AI/player 2), False for minimizing player (player 1)
    # - max_depth: maximum depth to search
    # Returns (best_score, best_action)
    winner = board.get_winner()
    if winner == 2: 
        return 1000, None
    elif winner == 1: 
        return -1000, None
    elif depth == max_depth:
        return board.evaluate(), None
    
    actions = board.get_all_possible_actions(not is_maximizing) 
    
    if not actions:
        return board.evaluate(), None
    
    best_action = None
    
    if is_maximizing: 
        best_score = float('-inf')
        for action in actions:
            new_board = board.clone()
            if new_board.apply_action(action, not is_maximizing):
                score, _ = minimax(new_board, depth + 1, alpha, beta, False, max_depth)
                if score > best_score:
                    best_score = score
                    best_action = action
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break 
    else: 
        best_score = float('inf')
        for action in actions:
            new_board = board.clone()
            if new_board.apply_action(action, not is_maximizing):
                score, _ = minimax(new_board, depth + 1, alpha, beta, True, max_depth)
                if score < best_score:
                    best_score = score
                    best_action = action
                beta = min(beta, best_score)
                if beta <= alpha:
                    break  
    
    return best_score, best_action

def ai_turn(board, difficulty=2):
    #AI turn using minimax with alpha-beta pruning
    print("AI is thinking...")
    start_time = time.time()
    
    
    _, best_action = minimax(board, 0, float('-inf'), float('inf'), True, max_depth=difficulty)
    
    if best_action:
        action_type, action_data = best_action
        if action_type == 'move':
            board.move_player(action_data, is_player1=False)
            print(f"AI moved to {action_data}")
        elif action_type == 'wall':
            orientation, row, col = action_data
            board.place_wall(orientation, row, col, is_player1=False)
            print(f"AI placed a wall at {(row, col)}, orientation: {orientation}")
    
    end_time = time.time()
    print(f"AI took {end_time - start_time:.2f} seconds to decide")

def draw_instructions(board, wall_mode, orientation, difficulty):
    font = pygame.font.SysFont(None, 24)
    instructions = [
        "SPACE: Toggle wall/place mode",
        "H: Horizontal wall   |   V: Vertical wall",
        f"Mode: {'Wall' if wall_mode else 'Move'}   |   Orientation: {orientation.upper()}",
        f"P1 Walls Left: {board.p1_walls} | P2 Walls Left: {board.p2_walls}",
        f"AI Difficulty: {difficulty} (1-3, press 1/2/3 to change)"
    ]
    for i, text in enumerate(instructions):
        txt_surface = font.render(text, True, BLACK)
        screen.blit(txt_surface, (10, HEIGHT - 120 + i * 20))

def draw_board(board, wall_mode, orientation, difficulty):
    screen.fill(GREY)
    
    
    for i in range(ROWS):
        for j in range(COLS):
            rect = pygame.Rect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

    
    for (r, c) in board.h_walls:
        pygame.draw.rect(screen, BROWN, (c*CELL_SIZE, (r+1)*CELL_SIZE - WALL_THICKNESS//2, CELL_SIZE*2, WALL_THICKNESS))
    
    
    for (r, c) in board.v_walls:
        pygame.draw.rect(screen, BROWN, ((c+1)*CELL_SIZE - WALL_THICKNESS//2, r*CELL_SIZE, WALL_THICKNESS, CELL_SIZE*2))

    
    p1_row, p1_col = board.p1_pos
    p2_row, p2_col = board.p2_pos
    p1_x, p1_y = p1_col*CELL_SIZE + 10, p1_row*CELL_SIZE + 10
    p2_x, p2_y = p2_col*CELL_SIZE + 10, p2_row*CELL_SIZE + 10
    pygame.draw.ellipse(screen, BLUE, (p1_x, p1_y, CELL_SIZE-20, CELL_SIZE-20))
    pygame.draw.ellipse(screen, RED, (p2_x, p2_y, CELL_SIZE-20, CELL_SIZE-20))

    
    if wall_mode:
        x, y = pygame.mouse.get_pos()
        row, col = y // CELL_SIZE, x // CELL_SIZE
        if row < 8 and col < 8:
            if orientation == 'h':
                preview_rect = pygame.Rect(col*CELL_SIZE, (row+1)*CELL_SIZE - WALL_THICKNESS//2, 
                                       CELL_SIZE*2, WALL_THICKNESS)
            else:  # 'v'
                preview_rect = pygame.Rect((col+1)*CELL_SIZE - WALL_THICKNESS//2, row*CELL_SIZE,
                                       WALL_THICKNESS, CELL_SIZE*2)
            pygame.draw.rect(screen, (100, 100, 100, 128), preview_rect)

    draw_instructions(board, wall_mode, orientation, difficulty)
    pygame.display.flip()

def display_winner(winner):
    font = pygame.font.SysFont(None, 48)
    text = font.render(f"{winner} Wins!", True, RED if winner == "AI" else BLUE)
    screen.blit(text, (WIDTH // 2 - 100, HEIGHT // 2))
    pygame.display.flip()

def main():
    board = Board()
    run = True
    is_player1_turn = True
    wall_mode = False
    orientation = 'h'
    difficulty = 2

    while run:
        clock.tick(FPS)
        draw_board(board, wall_mode, orientation, difficulty)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    wall_mode = not wall_mode
                elif event.key == pygame.K_h:
                    orientation = 'h'
                elif event.key == pygame.K_v:
                    orientation = 'v'
                elif event.key == pygame.K_d:
                    board.debug_walls()
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3]:
                    difficulty = int(event.unicode)
                    print(f"AI difficulty set to {difficulty}")

            elif event.type == pygame.MOUSEBUTTONDOWN and is_player1_turn:
                x, y = pygame.mouse.get_pos()
                row, col = y // CELL_SIZE, x // CELL_SIZE

                if wall_mode:
                    if board.place_wall(orientation, row, col, is_player1=True):
                        is_player1_turn = False
                        print(f"Player placed a {orientation} wall at {row},{col}")
                    else:
                        print("Invalid wall placement!")
                else:
                    possible_moves = board.get_possible_moves(board.p1_pos)
                    if (row, col) in possible_moves:
                        board.move_player((row, col), is_player1=True)
                        is_player1_turn = False
                        print(f"Player moved to {row},{col}")
                    else:
                        print("Invalid move! Possible moves:", possible_moves)

        if board.game_over():
            draw_board(board, wall_mode, orientation, difficulty)
            winner = "You" if board.p1_pos[0] == 8 else "AI"
            display_winner(winner)
            pygame.time.wait(3000)
            break

        if not is_player1_turn and run:
            pygame.time.wait(500)
            ai_turn(board, difficulty)
            if board.game_over():
                draw_board(board, wall_mode, orientation, difficulty)
                display_winner("AI")
                pygame.time.wait(3000)
                break
            is_player1_turn = True

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
