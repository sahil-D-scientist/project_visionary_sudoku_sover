from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image
import base64

from all_func import (
    find_largest_contour,
    get_contour_corners,
    four_point_transform,
    remove_lines_and_thin_digits,
    extract_digits_from_warped,
    recognize_digits,
    get_solved_sudoku,
    overlay_digits_on_sudoku,pre_process_image
)
import threading
import time

app = Flask(__name__)
def periodic_logging():
    while True:
        with open("server_logs.txt", "a") as log_file:
            log_file.write(f"Server is live at {time.ctime()}\n")
        print(f"Server is live at {time.ctime()}")  # For console visibility
        time.sleep(300)  # Wait for 5 minutes

# Start the periodic logging in a background thread
log_thread = threading.Thread(target=periodic_logging, daemon=True)
log_thread.start()
@app.route("/", methods=["GET", "POST"])
def sudoku():
    if request.method == "GET":
        return render_template("index.html")
    else:
        if 'image' not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Convert the uploaded file into bytes
        file_bytes = BytesIO(file.read())

        # Open the image using PIL and convert it to a format compatible with OpenCV
        pil_image = Image.open(file_bytes).convert("RGB")
        image = np.array(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            # Preprocess the image and find the Sudoku grid
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            binary=pre_process_image(gray)

            
            # Find and process the largest contour
            largest_contour = find_largest_contour(binary)
            corners = get_contour_corners(largest_contour)
            warped = four_point_transform(image, corners)
            cleaned_image = remove_lines_and_thin_digits(warped)

            # Segment the grid into individual digit cells
            digit_cells = extract_digits_from_warped(cleaned_image)

            # Load the pre-trained CNN model
            model_path = "trained_model.h5"
            cnn_model = load_model(model_path)

            # Recognize the digits in the grid
            recognized_grid = recognize_digits(digit_cells, cnn_model)
            recognized_grid = [[int(cell) for cell in row] for row in recognized_grid]
            print(recognized_grid)

            # Solve the Sudoku puzzle
            solved_grid = get_solved_sudoku(recognized_grid)
            solved_grid = [[int(cell) for cell in row] for row in solved_grid]

            # Overlay the solved grid back onto the original image
            solved_image = overlay_digits_on_sudoku(image, recognized_grid, solved_grid)

            # Encode the solved image to Base64 for easy passing to the frontend
            _, buffer = cv2.imencode('.png', solved_image)
            solved_image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Return the Base64 str to the frontend
            return jsonify({"solved_image": solved_image_base64})
        
        except Exception as e:
            return jsonify({"error": f"An error occurred during processing: {str(e)}"}), 500



if __name__=="__main__":

    app.run(port=8080,debug=True)