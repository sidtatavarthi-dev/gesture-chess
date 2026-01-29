# Import required libraries
import cv2  # OpenCV for camera capture
import mediapipe as mp  # MediaPipe for hand tracking
import chess  # Python-chess library for chess logic
import pygame  # Pygame for graphics and display
import math  # Math functions for distance calculations
import numpy as np  # NumPy for array operations
import urllib.request  # For downloading the hand tracking model
import os  # For file system operations

# Initialize pygame system
pygame.init()

# Window and board dimensions
WINDOW_WIDTH = 1200  # Total window width
WINDOW_HEIGHT = 800  # Total window height
BOARD_SIZE = 640  # Chess board size (640x640 pixels)
SQUARE_SIZE = BOARD_SIZE // 8  # Each square is 80x80 pixels
CAMERA_WIDTH = 480  # Camera feed width
CAMERA_HEIGHT = 640  # Camera feed height

# Color definitions (RGB format)
WHITE = (240, 217, 181)  # Light tan for white squares
BLACK = (181, 136, 99)  # Dark tan for black squares
HIGHLIGHT = (186, 202, 68)  # Yellow-green for valid moves
SELECTED = (246, 246, 105)  # Bright yellow for selected square
CURSOR_COLOR = (255, 0, 0)  # Red for cursor

# Set up MediaPipe hand tracking options
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Configure hand tracking parameters
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),  # Path to model file
    running_mode=VisionRunningMode.VIDEO,  # Video mode for real-time tracking
    num_hands=1,  # Track only one hand
    min_hand_detection_confidence=0.5,  # Minimum confidence to detect hand
    min_hand_presence_confidence=0.5,  # Minimum confidence hand is present
    min_tracking_confidence=0.5  # Minimum confidence for tracking
)

# Create new chess board in starting position
board = chess.Board()

# Set up pygame display window
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Gesture Chess")  # Window title
clock = pygame.time.Clock()  # Clock for frame rate control

# Initialize webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Game state variables
cursor_x = 0  # Cursor x position
cursor_y = 0  # Cursor y position
selected_square = None  # Currently selected chess square
is_pinching = False  # Current pinch state
was_pinching = False  # Previous pinch state (to detect pinch start/end)
frame_count = 0  # Frame counter for MediaPipe

# Chess piece symbols (not used with letter display)
PIECE_SYMBOLS = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
}

def calculate_distance(point1, point2):
    # Calculate distance between two hand landmarks using Pythagorean theorem
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def detect_pinch(hand_landmarks):
    # Check if user is pinching (thumb and index finger touching)
    thumb_tip = hand_landmarks[4]  # Landmark 4 is thumb tip
    index_tip = hand_landmarks[8]  # Landmark 8 is index finger tip
    
    # Calculate distance between thumb and index finger
    distance = calculate_distance(thumb_tip, index_tip)
    
    # Return True if distance is small (fingers are close together)
    return distance < 0.05

def get_cursor_position(hand_landmarks):
    # Get screen position from index finger tip position
    index_tip = hand_landmarks[8]  # Index finger tip landmark
    
    # Map coordinates directly to board size (no flip)
    x = index_tip.x * BOARD_SIZE
    y = index_tip.y * BOARD_SIZE
    return int(x), int(y)

def coords_to_square(x, y):
    # Convert pixel coordinates to chess square number
    # Returns None if coordinates are outside the board
    if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
        return None
    
    # Calculate file (column a-h = 0-7)
    file = int(x // SQUARE_SIZE)
    # Calculate rank (row 1-8 = 0-7, flipped because y increases downward)
    rank = 7 - int(y // SQUARE_SIZE)
    
    # Return square number if valid
    if 0 <= file < 8 and 0 <= rank < 8:
        return chess.square(file, rank)
    return None

def draw_board():
    # Draw the chess board with alternating colors
    for rank in range(8):  # Loop through rows
        for file in range(8):  # Loop through columns
            # Alternate colors based on position
            color = WHITE if (rank + file) % 2 == 0 else BLACK
            
            # Create rectangle for this square
            rect = pygame.Rect(file * SQUARE_SIZE, (7 - rank) * SQUARE_SIZE, 
                             SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, color, rect)
            
            # Get chess square number
            square = chess.square(file, rank)
            
            # Highlight selected square in bright yellow
            if selected_square == square:
                pygame.draw.rect(screen, SELECTED, rect)
            elif selected_square is not None:
                # Highlight valid moves with green outline
                move = chess.Move(selected_square, square)
                if move in board.legal_moves:
                    pygame.draw.rect(screen, HIGHLIGHT, rect, 5)

def draw_pieces():
    # Draw chess pieces as letters on the board
    font = pygame.font.Font(None, 60)  # Font size 60
    
    # Letter representation for each piece type
    PIECE_LETTERS = {
        'P': 'P', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K',  # White pieces
        'p': 'p', 'n': 'n', 'b': 'b', 'r': 'r', 'q': 'q', 'k': 'k'   # Black pieces
    }
    
    # Loop through all 64 squares
    for square in chess.SQUARES:
        piece = board.piece_at(square)  # Get piece at this square
        if piece:
            # Get file and rank for position
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            
            # Get letter for this piece
            symbol = PIECE_LETTERS.get(piece.symbol(), piece.symbol())
            
            # White pieces = white text, Black pieces = black text
            if piece.color == chess.WHITE:
                text = font.render(symbol.upper(), True, (255, 255, 255))
            else:
                text = font.render(symbol.lower(), True, (0, 0, 0))
            
            # Center the letter in the square
            x = file * SQUARE_SIZE + SQUARE_SIZE // 2 - text.get_width() // 2
            y = (7 - rank) * SQUARE_SIZE + SQUARE_SIZE // 2 - text.get_height() // 2
            
            # Draw the piece
            screen.blit(text, (x, y))

def draw_cursor(x, y):
    # Draw red and white cursor circles at finger position
    pygame.draw.circle(screen, CURSOR_COLOR, (x, y), 10, 3)  # Red outer circle
    pygame.draw.circle(screen, (255, 255, 255), (x, y), 8, 2)  # White inner circle

def draw_camera_feed(frame):
    # Display camera feed on the right side of the screen
    # Convert BGR (OpenCV format) to RGB (pygame format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize, rotate, and convert to pygame surface
    frame_surface = pygame.surfarray.make_surface(
        np.rot90(cv2.resize(frame_rgb, (CAMERA_WIDTH, CAMERA_HEIGHT)))
    )
    # Draw camera feed to the right of the board
    screen.blit(frame_surface, (BOARD_SIZE + 20, 0))

def draw_info():
    # Draw game information text
    font = pygame.font.Font(None, 36)
    
    # Show whose turn it is
    turn_text = "White's Turn" if board.turn == chess.WHITE else "Black's Turn"
    text = font.render(turn_text, True, (255, 255, 255))
    screen.blit(text, (BOARD_SIZE + 20, CAMERA_HEIGHT + 20))
    
    # Show pinch status (green when pinching)
    pinch_text = "PINCHING" if is_pinching else "Open Hand"
    color = (0, 255, 0) if is_pinching else (255, 255, 255)
    text = font.render(pinch_text, True, color)
    screen.blit(text, (BOARD_SIZE + 20, CAMERA_HEIGHT + 60))
    
    # Show game status (check or checkmate)
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        text = font.render(f"Checkmate! {winner} Wins!", True, (255, 0, 0))
        screen.blit(text, (BOARD_SIZE + 20, CAMERA_HEIGHT + 100))
    elif board.is_check():
        text = font.render("Check!", True, (255, 165, 0))
        screen.blit(text, (BOARD_SIZE + 20, CAMERA_HEIGHT + 100))

# Download hand tracking model if not already present
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading hand tracking model...")
    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded!")

# Create hand tracking object
landmarker = HandLandmarker.create_from_options(options)

# Main game loop - runs until user quits
running = True
while running:
    # Check for user input events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # X button clicked
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:  # ESC key pressed
                running = False
            elif event.key == pygame.K_r:  # R key pressed - reset game
                board.reset()
                selected_square = None
    
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:  # Skip if frame capture failed
        continue
    
    frame_count += 1  # Increment frame counter
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert frame to RGB format for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    # Detect hands in the frame
    results = landmarker.detect_for_video(mp_image, frame_count)
    
    # If hand was detected, process gestures
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Update cursor position based on index finger
            cursor_x, cursor_y = get_cursor_position(hand_landmarks)
            
            # Check if user is pinching
            is_pinching = detect_pinch(hand_landmarks)
            
            # Handle pinch start (select piece)
            if is_pinching and not was_pinching:
                # User just started pinching - try to select a piece
                square = coords_to_square(cursor_x, cursor_y)
                if square is not None:
                    piece = board.piece_at(square)
                    # Only select if it's the current player's piece
                    if piece and piece.color == board.turn:
                        selected_square = square
                        print(f"Selected: {chess.square_name(square)}")
            
            # Handle pinch release (move piece)
            elif not is_pinching and was_pinching:
                # User just released pinch - try to move the piece
                if selected_square is not None:
                    target_square = coords_to_square(cursor_x, cursor_y)
                    if target_square is not None:
                        move = chess.Move(selected_square, target_square)
                        
                        # Handle pawn promotion (auto-promote to queen)
                        piece = board.piece_at(selected_square)
                        if piece and piece.piece_type == chess.PAWN:
                            if (piece.color == chess.WHITE and chess.square_rank(target_square) == 7) or \
                               (piece.color == chess.BLACK and chess.square_rank(target_square) == 0):
                                move = chess.Move(selected_square, target_square, promotion=chess.QUEEN)
                        
                        # Make the move if it's legal
                        if move in board.legal_moves:
                            board.push(move)
                            print(f"Moved: {chess.square_name(selected_square)} to {chess.square_name(target_square)}")
                            selected_square = None
                        else:
                            print("Illegal move!")
                            selected_square = None
                    else:
                        selected_square = None
            
            # Update previous pinch state
            was_pinching = is_pinching
    
    # Draw everything to screen
    screen.fill((50, 50, 50))  # Dark gray background
    draw_board()  # Draw chess board
    draw_pieces()  # Draw chess pieces
    draw_cursor(cursor_x, cursor_y)  # Draw cursor
    draw_camera_feed(frame)  # Draw camera feed
    draw_info()  # Draw game info
    
    # Update display
    pygame.display.flip()
    clock.tick(30)  # Limit to 30 FPS

# Clean up resources when game ends
cap.release()  # Release webcam
landmarker.close()  # Close hand tracker
pygame.quit()  # Quit pygame
