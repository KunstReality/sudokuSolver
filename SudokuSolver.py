import math

import numpy as np
import cv2
import imutils


def show_image(img):
    cv2.imshow('image', img)  # Display the image
    cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
    cv2.destroyAllWindows()  # Close all windows

def display_points(in_img, points, color=(0, 0, 255)):
    img = in_img.copy()
    for point in points:
        img = cv2.drawContours(img, [point], -1, color, 3)
    show_image(img)
    return img


def display_rects(in_img, contours, color=255):
    img = in_img.copy()
    for rect in contours:
        img = cv2.drawContours(img, [rect], -1, color, 3)
    show_image(img)
    return img

def display_lines(in_img, lines, color=(0, 0, 255)):
    img = in_img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        img = cv2.line(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    show_image(img)
    return img

#Blur img to reduce Noise
def preprocess_img(img):
    # get the gray img
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply a bilateral filter to smooth out contrast
    # it is also possible to use a Gaussian Filter here
    # @params: d, sigma color, sigma Space
    gauss_img = cv2.GaussianBlur(img_gray, (9, 9), 0)
    threshold = cv2.adaptiveThreshold(gauss_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], dtype=np.uint8)
    erosion_img = cv2.erode(threshold, kernel)
    return erosion_img

def locate_sudoku(img):
    # find edges using Canny
    # @params: first and second Threshold
    edges = cv2.Canny(img, 50, 150)
    # find the vertices of two contour.
    vertices = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(vertices)

    # sort contours based on the Area of a contour from big to small
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # find the actual board in the img
    board = None
    for contour in sorted_contours:
        approximated_poly = cv2.approxPolyDP(contour, 10, True)
        if len(approximated_poly) == 4:
            #board = approximated_poly
            rect = cv2.minAreaRect(approximated_poly)
            box = cv2.boxPoints(rect)
            boards_list = []
            for i in range(len(box)):
                boards_list.append(np.array([box[i]], dtype=np.int32))
            board = np.array(boards_list, dtype=np.int32)
            break
    return board

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

# Projekt the board into a 900 * 900 window
def get_perspektiveProjection(img, board, height=900, width=900):
    board = sort_points(board)
    board_vertecies = np.float32([board[0], board[1], board[2], board[3]])
    box_vertecies = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

    projektMatrix = cv2.getPerspectiveTransform(board_vertecies, box_vertecies)
    return cv2.warpPerspective(img, projektMatrix, (height, width))

def find_lines(img):
    gauss_img = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(gauss_img, 60, 150)
    show_image(edges)
    minLineLength = 100
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=minLineLength, maxLineGap=80)
    return lines


def main():
    for i in range(1, 9):
        img = cv2.imread('sudoku' + str(i) + '.jpg')
        processed_img = preprocess_img(img)
        board = locate_sudoku(processed_img)
        if board is not None:
            perspective_img = get_perspektiveProjection(img, board)
            lines = find_lines(perspective_img)
            if lines is not None:
                display_lines(perspective_img, lines)


if __name__ == '__main__':
    main()
