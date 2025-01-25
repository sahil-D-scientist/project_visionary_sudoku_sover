
# Visionary Sudoku Solver

Visionary Sudoku Solver is a Python-based project designed to solve Sudoku puzzles from an image using computer vision and deep learning techniques. It features a Flask web interface for users to upload Sudoku images, process them, and solve puzzles in real-time.

---

## Features
- Detects and extracts Sudoku grids from images using OpenCV.
- Recognizes digits in the grid using a pre-trained deep learning model.
- Solves the Sudoku puzzle and overlays the solution back onto the image.
- Simple and interactive web interface powered by Flask.

---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/visionary-sudoku-solver.git
cd visionary-sudoku-solver
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Install all required Python packages using `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Start the Flask application:
```bash
python app.py
```

### 5. Access the Application
Open your browser and navigate to:
```
http://127.0.0.1:8080
```

---

## Deployment

1. **Ensure all dependencies are installed using the `requirements.txt` file.**
2. **Start the application by running:**
   ```bash
   python app.py
   ```
3. **Host the application on a live server for public access if required.**

---

## Live Project
You can find the live project [here](#https://project-visionary-sudoku-solver.onrender.com).
