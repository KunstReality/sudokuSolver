# Sudoku Solver
this is an automatic Sudoku puzzle solver using OpenCV, Tensorflow
and Optical Character Recognition (OCR).
## Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Usage](#usage)
4. [Contributing](#contributing)
5. [License](#license)

## Introduction
This Python script is designed to solve Sudoku puzzles using computer vision and machine learning techniques. 
With this tool, you can easily solve Sudoku puzzles from live images from your phone camera.

## Requirements
Before using the Sudoku Solver, make sure you have the following dependencies installed:

- **OpenCV**: Open Source Computer Vision Library. You can install it using pip:
   ```bash
   pip install opencv-python
- **Py-Sudoku**: Python library for solving Sudoku puzzles. Install it with pip:
  ```bash  
    pip install py-sudoku

**TensorFlow**: An open-source machine learning framework. [You can find installation instructions in the TensorFlow Installation Guide provided by TensorFlow.](https://www.tensorflow.org/install/pip)


**Droidcam**: This project requires Droidcam to access a webcam feed. [Follow the setup instructions in the Droidcam Simple Setup documentation to install and configure Droidcam on your system.](https://github.com/cardboardcode/droidcam_simple_setup)

## Usage
To use the Sudoku Solver script, follow these steps:

1. **Clone this repository** to your local machine:

   ```bash
   git clone https://github.com/KunstReality/sudokuSolver.git

Replace yourusername with your actual GitHub username or provide the repository URL.

2. **Navigate to the project directory:**
    ```bash
    cd sudokuSolver
   
3. **Open the Droidcam-App on your phone.**
4. **Run the Sudoku Solver script** with the following command:
    ```bash
    python SudokuSolver -ip "ip Address"
Replace "ip Address" with the IP address which the **Droidcam-App** suggest.

