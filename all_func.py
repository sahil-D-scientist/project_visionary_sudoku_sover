

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

def find_largest_contour(image):
    """Finds the largest contour in a binary image."""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def get_contour_corners(contour):
    """Approximates the corners of a contour."""
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx_corners = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx_corners) == 4:
        return approx_corners.reshape((4, 2))
    else:
        raise ValueError("Could not find exactly 4 corners.")

def order_points(pts):
    """Orders the points in a consistent way: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect

def four_point_transform(image, pts):
    """Performs a perspective transformation based on four points."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width and height of the new image
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    # Define the destination points for a "birds-eye" view
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix and apply it
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped



def pre_process_image_new(img, skip_dilate=False, flag=0):
    """Uses a blurring function, adaptive thresholding and dilation to expose the main features of an image."""

    # Gaussian blur with a kernal size (height, width) of 9.
    # Note that kernal sizes must be positive and odd and the kernel must be square.
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)

    # Adaptive threshold using 11 nearest neighbour pixels
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)

    # Invert colours, so gridlines have non-zero pixel values.
    # Necessary to dilate the image, otherwise will look like erosion instead.
    proc = cv2.bitwise_not(proc, proc)

    if not skip_dilate:
        # Dilate the image to increase the size of the grid lines.
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        proc = cv2.dilate(proc, kernel)

    return proc
def remove_lines_and_thin_digits(image):
    """
    Aggressively remove horizontal and vertical lines from the Sudoku grid, and preserve digits.
    """
    # Step 1: Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    binary=pre_process_image_new(gray,skip_dilate=True)
    # cv2.imshow("Step 3: Binary Thresholding", binary)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # Step 4: Create horizontal and vertical kernels
    height, width = binary.shape
    horizontal_kernel_size = max(20, width // 16)
    vertical_kernel_size = max(20, height // 15)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_size, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_size))

    # Step 5: Extract horizontal lines
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    # cv2.imshow("Step 5: Horizontal Lines", horizontal_lines)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # Step 6: Extract vertical lines
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    # cv2.imshow("Step 6: Vertical Lines", vertical_lines)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # Step 7: Combine and dilate lines
    combined_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
    dilated_lines = cv2.dilate(combined_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    # cv2.imshow("Step 7: Dilated Lines", dilated_lines)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # Step 8: Remove lines from binary image
    no_lines = cv2.bitwise_and(binary, cv2.bitwise_not(dilated_lines))
    # cv2.imshow("Step 8: No Lines Image", no_lines)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # Step 9: Find and filter contours to clean small artifacts
    contours, _ = cv2.findContours(no_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area_threshold = 90
    mask = np.zeros_like(no_lines)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area_threshold:
            cv2.drawContours(mask, [cnt], -1, 255, -1)
    # cv2.imshow("Step 9: Mask for Digits", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # Step 10: Clean binary image using the mask
    cleaned_binary = cv2.bitwise_and(no_lines, mask)
    # cv2.imshow("Step 10: Cleaned Binary", cleaned_binary)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # Step 11: Invert the image
    cleaned_image = cv2.bitwise_not(cleaned_binary)
    # cv2.imshow("Step 11: Cleaned Image", cleaned_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # Step 12: Inpaint the cleaned image to restore digits
    inpainted = cv2.inpaint(cleaned_image, dilated_lines, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    # cv2.imshow("Step 12: Final Inpainted Image", inpainted)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)

    return inpainted

# def plot_image(image, title="Image"):
#     """Plots a single image using Matplotlib."""
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.title(title)
#     plt.axis("off")
#     plt.show()

def pre_process_image(img, skip_dilate=False, flag=0):
    """Uses a blurring function, adaptive thresholding and dilation to expose the main features of an image."""

    # Gaussian blur with a kernal size (height, width) of 9.
    # Note that kernal sizes must be positive and odd and the kernel must be square.
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)

    # Adaptive threshold using 11 nearest neighbour pixels
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert colours, so gridlines have non-zero pixel values.
    # Necessary to dilate the image, otherwise will look like erosion instead.
    proc = cv2.bitwise_not(proc, proc)

    if not skip_dilate:
        # Dilate the image to increase the size of the grid lines.
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        proc = cv2.dilate(proc, kernel)

    return proc


# def plot_image(image, title="Image"):
#     """Plots a single image using Matplotlib."""
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.title(title)
#     plt.axis("off")
#     plt.show()

def extract_digits_from_warped(warped_grid):
    """
    Extracts digits from the already warped Sudoku grid and stores them in a 2D list,
    adjusting the area upwards and downwards for better digit extraction.
    """
    # Ensure the input is grayscale
    if len(warped_grid.shape) == 3:
        warped_grid = cv2.cvtColor(warped_grid, cv2.COLOR_BGR2GRAY)

    # Get the dimensions of the warped grid
    height, width = warped_grid.shape

    # Dynamically compute the cell size
    cell_height = height // 9
    cell_width = width // 9

    # Initialize the 2D list for digits
    digit_grid = []

    # Adjustment parameters (change these values as needed)
    upward_adjustment = 0.05  # Decrease area upwards by 10% of cell height
    downward_adjustment = 0.05  # Increase area downwards by 20% of cell height

    # Loop through rows
    for i in range(9):
        row = []
        # Loop through columns
        for j in range(9):
            # Calculate adjusted coordinates
            start_y = int(i * cell_height + cell_height * upward_adjustment)
            end_y = int((i + 1) * cell_height + cell_height * downward_adjustment)
            start_x = j * cell_width
            end_x = (j + 1) * cell_width

            # Ensure boundaries stay within the grid dimensions
            start_y = max(0, start_y)
            end_y = min(height, end_y)
            start_x = max(0, start_x)
            end_x = min(width, end_x)

            # Extract the adjusted cell using slicing
            cell = warped_grid[start_y:end_y, start_x:end_x]

            # Append the adjusted cell to the current row
            row.append(cell)

        # Append the row to the digit grid
        digit_grid.append(row)

    # Return the 2D digit grid
    return digit_grid


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
    
    # Display the preprocessed digit (optional)
    # cv2.imshow("Preprocessed Digit", digit_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    
    # Normalize pixel values to the range [0, 1]
    digit_image = digit_image.astype("float32") / 255.0  

    # Add the required dimensions for the model input
    digit_image = np.expand_dims(digit_image, axis=-1)  # Add channel dimension (grayscale)
    digit_image = np.expand_dims(digit_image, axis=0)   # Add batch dimension
    
    return digit_image
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

                    # Mark the digit as 0 if confidence is less than 0.5
                    if confidence < 0.5:
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
    """
    Overlay solved digits onto the Sudoku grid and return the final output image.
    """
    # Load the image
    if image is None:
        raise ValueError("Error: Image could not be loaded.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocess the image (blurring + thresholding)
    binary = pre_process_image(gray)

    # Find the largest contour
    try:
        largest_contour = find_largest_contour(binary)
        corners = get_contour_corners(largest_contour)
    except ValueError as e:
        raise ValueError(f"Error: {e}")

    # Perform a perspective transform
    warped = four_point_transform(image, corners)

    # Define font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    # Calculate cell dimensions dynamically
    grid_height, grid_width = warped.shape[:2]
    cell_height = grid_height // 9
    cell_width = grid_width // 9

    # Overlay solved digits
    for i in range(9):
        for j in range(9):
            if initial_grid[i][j] == 0:  # Only overlay solved cells
                digit = solved_grid[i][j]

                # Calculate the size of the digit text
                text_size = cv2.getTextSize(str(digit), font, font_scale, thickness)[0]

                # Calculate the center of the current cell
                cell_center_x = j * cell_width + cell_width // 2
                cell_center_y = i * cell_height + cell_height // 2

                # Calculate the top-left position for the digit text
                text_x = cell_center_x - text_size[0] // 2  # Center horizontally
                text_y = cell_center_y + text_size[1] // 2  # Center vertically

                # Draw the text on the warped image
                cv2.putText(warped, str(digit), (text_x, text_y), font, font_scale, (255, 0, 0), thickness)

    # Inverse the perspective transform
    rect = order_points(corners)
    dst = np.array([
        [0, 0],
        [grid_width - 1, 0],
        [grid_width - 1, grid_height - 1],
        [0, grid_height - 1]
    ], dtype="float32")

    inverse_matrix = cv2.getPerspectiveTransform(dst, rect)
    inversed = cv2.warpPerspective(warped, inverse_matrix, (image.shape[1], image.shape[0]))

    # Blend the inversed image with the original
    mask = cv2.warpPerspective(np.ones_like(warped, dtype=np.uint8) * 255, inverse_matrix, (image.shape[1], image.shape[0]))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    masked_image = cv2.bitwise_and(image, image, mask=255 - mask)
    output_image = cv2.add(masked_image, inversed)

    # Return the final output image
    return output_image