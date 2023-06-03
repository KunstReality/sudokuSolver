import math

import numpy as np
import imutils
from OpenCVHelpers import *

"""
Blur img to reduce Noise. This function uses a gaussian blur for reducing noise
and uses a binary Threshold which splits the img to 0/1 values
"""


def preprocess_img(img, skip_morph=False):
    # converts color img to gray img
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gaussian using kernel size of (.., ..)
    gauss_img = cv2.GaussianBlur(img_gray, (11, 11), 0)

    # cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, constant(c))
    threshold = cv2.adaptiveThreshold(gauss_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 1)

    # edges will have a non zero value
    threshold = cv2.bitwise_not(threshold, threshold)

    if not skip_morph:
        # perform morphology to make edge detection more robust
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], dtype=np.uint8)
        threshold = cv2.dilate(threshold, kernel)
        threshold = cv2.erode(threshold, kernel)

    return threshold


def locate_sudoku(img):
    # find edges using Canny
    # @params: first and second Threshold
    edges = cv2.Canny(img, 200, 250)

    # findContours: boundaries of shapes having same intensity
    vertices = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(vertices)

    # sort contours based on the Area of a contour from big to small
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # find the actual board in the img
    board = None
    for contour in sorted_contours:
        approximated_poly = cv2.approxPolyDP(contour, 50, True)
        if len(approximated_poly) >= 4:
            if math.fabs(cv2.contourArea(approximated_poly)) > 1000:
                rect = cv2.minAreaRect(approximated_poly)
                box = cv2.boxPoints(rect)
                point_list = []
                for i in range(len(box)):
                    point_list.append(np.array([box[i]], dtype=np.int32))
                board = np.array(point_list, dtype=np.int32)
                break
    return board

"""
brings the board points in the same order left-up left-down right-up righ-down
"""
def sort_points(board):
    sorted_board = np.zeros_like(board)

    sorted_x = board[np.argsort(board[:, 0, 0]), :]

    lefts = sorted_x[:2, :, :]
    rights = sorted_x[2:, :, :]

    sorted_lefts = lefts[np.argsort(lefts[:, 0, 1]), :]
    sorted_rights = rights[np.argsort(rights[:, 0, 1]), :]

    sorted_board[0] = sorted_lefts[0]
    sorted_board[1] = sorted_lefts[1]
    sorted_board[2] = sorted_rights[1]
    sorted_board[3] = sorted_rights[0]

    return sorted_board


"""
Projekt the board into a 900 * 900 window
"""
def get_perspektiveProjection(img, board, height=900, width=900):
    board = sort_points(board)
    board_vertecies = np.float32([board[0], board[1], board[2], board[3]])
    box_vertecies = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

    projektMatrix = cv2.getPerspectiveTransform(board_vertecies, box_vertecies)
    return cv2.warpPerspective(img, projektMatrix, (height, width))

# deprecated
def find_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 9)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

    edges = cv2.Canny(opening, 100, 150)
    minLineLength = 100
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=minLineLength, maxLineGap=80)
    return lines


def locate_cells(img):

    process_img = preprocess_img(img, False)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    opening = cv2.morphologyEx(process_img, cv2.MORPH_OPEN, kernel, iterations=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel, iterations=3)

    show_image(opening)

    vertices = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(vertices)

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=False)
    # find the board cells
    cells = []
    for contour in sorted_contours:
        peri = cv2.arcLength(contour, True)
        approximated_poly = cv2.approxPolyDP(contour, 0.015 * peri, True)
        if len(approximated_poly) >= 4:
            if 3000 < math.fabs(cv2.contourArea(approximated_poly)) < 10000:
                x, y, w, h = cv2.boundingRect(approximated_poly)
                cells.append(np.array([x, y, x + w, y + h], dtype=np.int32))

    return np.array(cells, dtype=np.int32)


def main():
    for i in range(1, 9):
        img = cv2.imread('sudoku' + str(i) + '.jpg')
        print(i)
        processed_img = preprocess_img(img)
        board = locate_sudoku(processed_img)
        if board is not None:
            perspective_img = get_perspektiveProjection(img, board)

            cells = locate_cells(perspective_img)
            for cell in cells:
                cv2.rectangle(perspective_img, (cell[0], cell[1]), (cell[2], cell[3]), (36, 255, 12), 3)
            show_image(perspective_img)


if __name__ == '__main__':
    main()
