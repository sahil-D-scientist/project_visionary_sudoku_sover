

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

import numpy as np

def is_valid(board, row, col, num):
    """
    Check if placing a number in a given position is valid.
    """
    # Check the row
    if num in board[row]:
        return False

    # Check the column
    if num in [board[i][col] for i in range(9)]:
        return False

    # Check the 3x3 subgrid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i][j] == num:
                return False

    return True

def solve_sudoku(board):
    """
    Solve the Sudoku puzzle using backtracking.
    """
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        board[row][col] = num

                        if solve_sudoku(board):
                            return True

                        # Backtrack
                        board[row][col] = 0

                return False

    return True

def get_solved_sudoku(board):
    """
    Solve the Sudoku puzzle and return the solved grid as a 2D list.
    """
    # Make a copy of the board to avoid modifying the original
    solved_board = [row[:] for row in board]

    if solve_sudoku(solved_board):
        return [list(row) for row in solved_board]  # Convert NumPy arrays to lists
    else:
        raise ValueError("No solution exists for the given Sudoku grid.")

def remove_lines_and_thin_digits(image):
    """
    Aggressively remove horizontal and vertical lines from the Sudoku grid, and preserve digits.
    """
    import cv2
    import numpy as np

    # Convert to grayscale if the image is not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply binary inverse thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Adaptive morphological kernel sizes based on image size
    height, width = binary.shape
    horizontal_kernel_size = max(25, width // 18)
    vertical_kernel_size = max(25, height // 18)

    # Create horizontal and vertical kernels
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_size, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_size))

    # Detect horizontal and vertical lines using morphology
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Combine the line masks
    combined_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)

    # Dilate the lines to ensure full coverage
    dilated_lines = cv2.dilate(combined_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

    # Remove lines from the binary image
    no_lines = cv2.bitwise_and(binary, cv2.bitwise_not(dilated_lines))

    # Remove small noise or dots using area filtering
    contours, _ = cv2.findContours(no_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area_threshold = 35  # Minimum area for contours to keep
    mask = np.zeros_like(no_lines)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area_threshold:
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    # Apply the mask to the cleaned binary image
    cleaned_binary = cv2.bitwise_and(no_lines, mask)

    # Invert the image back to the original polarity
    cleaned_image = cv2.bitwise_not(cleaned_binary)

    # Optional: Smooth the image using inpainting
    inpainted = cv2.inpaint(cleaned_image, dilated_lines, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return inpainted





def grid(image):
    """
    Detect the Sudoku grid using contours, perform perspective transform,
    and clean the grid using line removal.
    """
    # print(f"Loading image from: {image_path}")
    # image = cv2.imread(image_path)
    # if image is None:
    #     print("Error: Image not loaded. Please check the file path.")
    #     return None

    # Resize and preprocess the image
    image = cv2.resize(image, (450, 450))  # Resize to standard size
    original_image = image.copy()
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)  # Edge detection for grid detection

    # Find contours to detect the largest rectangle (grid)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found!")
        return None

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        # Perspective transform
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left

        # Transform the grid to a 450x450 image
        dst = np.array([[0, 0], [450, 0], [450, 450], [0, 450]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(original_image, M, (450, 450))

        # Clean the grid by removing lines
        cleaned_grid = remove_lines_and_thin_digits(warped)

        return cleaned_grid
    else:
        print("Sudoku grid not found!")
        return None
    



def segment_and_store_digits(warped_grid):
    """
    Segment the warped Sudoku grid into 9x9 cells and store each cell in a 2D list using np.vsplit and np.hsplit.
    """
    # Convert to grayscale if needed
    if len(warped_grid.shape) == 3:
        warped_grid = cv2.cvtColor(warped_grid, cv2.COLOR_BGR2GRAY)

    # Ensure the grid is a square and divisible by 9
    height, width = warped_grid.shape
    assert height == width, "The input grid must be square."
    assert height % 9 == 0, "The grid size must be divisible by 9."

    # Split the grid into 9 rows
    rows = np.vsplit(warped_grid, 9)
    digits = []

    # Split each row into 9 cells (columns) and append to the digits list
    for row in rows:
        cols = np.hsplit(row, 9)
        digits.append(cols)

    return digits


def preprocess_digit_for_prediction(digit_image):
    """
    Preprocess a single digit image for prediction by the CNN model.
    - Convert to grayscale if necessary.
    - Apply adaptive thresholding to convert to binary.
    - Resize to 32x32 (as per the model's input size).
    - Normalize pixel values to the range [0, 1].
    - Expand dimensions to match the input shape for the model.
    """
    # Convert to grayscale if the image is not already grayscale
    if len(digit_image.shape) == 3:  # If it has 3 channels (BGR)
        digit_image = cv2.cvtColor(digit_image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to create a binary image
    _, digit_image = cv2.threshold(
        digit_image, 
        127,    # Threshold value
        255,    # Maximum value to use with the THRESH_BINARY method
        cv2.THRESH_BINARY
    )
    
    # Resize to 32x32 (assumes your model takes 32x32 inputs)
    digit_image = cv2.resize(digit_image, (32, 32))  
    # cv2.imshow("digit",digit_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    
    # Normalize pixel values to the range [0, 1]
    digit_image = digit_image.astype("float32") / 255.0  

    # Add the required dimensions for the model input
    digit_image = np.expand_dims(digit_image, axis=-1)  # Add channel dimension (grayscale)
    digit_image = np.expand_dims(digit_image, axis=0)   # Add batch dimension
    
    return digit_image
import cv2
import numpy as np

def recognize_digits(digits,cnn_model):
    """
    Recognize digits in a Sudoku grid using a pre-trained model.
    - Checks for the percentage of content (non-zero pixels) in the digit image.
    - Deskews and preprocesses each digit image before recognition.
    - Returns the recognized grid as a 2D array.
    """
    sudoku_grid = []

    for row in digits:
        sudoku_row = []
        for digit in row:
            if digit is not None:
                try:
                    # Preprocess the digit (deskew and normalize)
                    digit = preprocess_digit_for_prediction(digit)

                    # Predict the digit
                    prediction = cnn_model.predict(digit, verbose=0)
                    recognized_digit = np.argmax(prediction)
                    confidence = np.max(prediction)  # Get the confidence score of the prediction

                    # Mark the digit as 0 if confidence is less than 0.6
                    if confidence < 0.6:
                        recognized_digit = 0

                    sudoku_row.append(recognized_digit)
                except Exception as e:
                    sudoku_row.append(0)  # Fallback to 0 for errors
            else:
                sudoku_row.append(0)  # Empty cell (None)
        sudoku_grid.append(sudoku_row)
    
    # Convert the list of lists into a 2D NumPy array
    sudoku_grid_array = np.array(sudoku_grid, dtype=int)

    return sudoku_grid_array


def overlay_digits_on_sudoku(image, initial_grid, solved_grid):
    # Load the Sudoku image
    # image = cv2.imread(image_path)

    if image is None:
        print("Error loading the image.")
        return

    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours and locate the largest rectangular contour (assumed to be the Sudoku grid)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    grid_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:  # Looking for a quadrilateral
                max_area = area
                grid_contour = approx

    if grid_contour is None:
        print("Could not detect the Sudoku grid.")
        return

    # Warp perspective to a flat view of the Sudoku grid
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        return rect

    ordered_points = order_points(grid_contour.reshape(4, 2))
    grid_points = np.float32([[0, 0], [450, 0], [450, 450], [0, 450]])
    matrix = cv2.getPerspectiveTransform(ordered_points, grid_points)
    warped = cv2.warpPerspective(image, matrix, (450, 450))

    # Define font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    cell_size = 450 // 9  # Assuming a 9x9 grid


    # Overlay solved digits
    for i in range(9):
        for j in range(9):
            if initial_grid[i][j] == 0:  # Only draw solved cells
                digit = solved_grid[i][j]
                text_size = cv2.getTextSize(str(digit), font, font_scale, thickness)[0]
                text_x = j * cell_size + (cell_size - text_size[0]) // 2
                text_y = i * cell_size + (cell_size + text_size[1]) // 2
                cv2.putText(warped, str(digit), (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

    # Map the modified grid back to the original perspective
    inverse_matrix = cv2.getPerspectiveTransform(grid_points, ordered_points)
    inversed = cv2.warpPerspective(warped, inverse_matrix, (image.shape[1], image.shape[0]))

    # Blend the inversed grid with the original image
    mask = cv2.warpPerspective(np.ones_like(warped) * 255, inverse_matrix, (image.shape[1], image.shape[0]))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    masked_image = cv2.bitwise_and(image, image, mask=255 - mask)
    output_image = cv2.add(masked_image, inversed)
    return output_image

