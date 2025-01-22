from flask import Flask, render_template, request, redirect,send_file,jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from all_func import grid, segment_and_store_digits, recognize_digits, overlay_digits_on_sudoku,get_solved_sudoku
from io import BytesIO
from PIL import Image
import base64


app=Flask(__name__)


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

        cleaned_image = grid(image)
        if cleaned_image is None:
            return jsonify({"error": "Failed to detect and clean the Sudoku grid."}), 500

        model_path = "/Users/sahilsingla/Downloads/deployment_series/flask/project_visionary_sudoku_solver/trained_model.h5"

        # Segment the grid into individual digit cells
        digit_cells = segment_and_store_digits(cleaned_image)

        # Load the pre-trained CNN model
        cnn_model = load_model(model_path)

        # Recognize the digits in the grid
        recognized_grid = recognize_digits(digit_cells, cnn_model)
        recognized_grid = [[int(cell) for cell in row] for row in recognized_grid]

        # Solve the Sudoku puzzle
        solved_grid = get_solved_sudoku(recognized_grid)
        solved_grid = [[int(cell) for cell in row] for row in solved_grid]

        # Overlay the solved grid back onto the original image
        solved_image = overlay_digits_on_sudoku(image, recognized_grid, solved_grid)

        # Encode the solved image to Base64 for easy passing to the frontend
        _, buffer = cv2.imencode('.png', solved_image)
        solved_image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Return the Base64 string to the frontend
        return jsonify({"solved_image": solved_image_base64})


if __name__=="__main__":

    app.run(port=8998,debug=True)