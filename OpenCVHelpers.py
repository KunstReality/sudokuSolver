import cv2
import numpy as np
import imutils
import math
from skimage.segmentation import clear_border


def show_image(img, name='image', wait=False):
    cv2.imshow(name, img)  # Display the image
    if wait:
        cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
    # cv2.destroyAllWindows()  # Close all windows


def display_points(in_img, points, color=(0, 0, 255)):
    img = in_img.copy()
    for point in points:
        img = cv2.drawContours(img, [point], -1, color, 3)
    return img


def display_rect(in_img, contours, color=255):
    img = in_img.copy()
    for rect in contours:
        img = cv2.drawContours(img, [rect], -1, color, 3)
    return img


def display_lines(in_img, lines, color=(0, 0, 255)):
    img = in_img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        img = cv2.line(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    return img


def extract_digit(cell, debug=False):
    # apply automatic thresholding to the cell and then clear any
    # connected borders that touch the border of the cell

    thresh = cv2.threshold(cell, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    thresh = clear_border(thresh)

    # check to see if we are visualizing the cell thresholding step
    if debug:
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)

    # find contours in the thresholded cell
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # if no contours were found than this is an empty cell
    if len(cnts) == 0:
        return 255 * np.ones((cell.shape))

    # otherwise, find the largest contour in the cell and create a
    # mask for the contour
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    # compute the percentage of masked pixels relative to the total
    # area of the image
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    # if less than 3% of the mask is filled then we are looking at
    # noise and can safely ignore the contour
    if percentFilled < 0.015:
        return 255 * np.ones((cell.shape))
    # apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    # check to see if we should visualize the masking step
    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)
    # return the digit to the calling function
    digit = cv2.bitwise_not(digit, digit)
    return digit


def resize_cell(box_img, img_size):
    return cv2.resize(box_img, (img_size, img_size)) / 255.0


def get_box_imgs(img, boxes, img_size, getDigits=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box_imgs = []
    for x, y, w, h in boxes:
        grayBox = gray[y + int(h / 7): y + h, x + int(w / 7): x + w]
        if getDigits:
            grayBox = extract_digit(grayBox)

        if grayBox is not None:
            grayBox = resize_cell(grayBox, img_size)
        box_imgs.append(grayBox)

    return box_imgs


def locate_cells(img):
    process_img = preprocess_img(img, False)
    # show_image(process_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    opening = cv2.morphologyEx(process_img, cv2.MORPH_OPEN, kernel, iterations=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel, iterations=3)

    vertices = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(vertices)
    # show_image(opening)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=False)
    # find the board cells
    cells = []
    for contour in sorted_contours:
        # peri = cv2.arcLength(contour, True)
        approximated_poly = cv2.approxPolyDP(contour, 10, True)
        if len(approximated_poly) >= 4:
            if 500 < math.fabs(cv2.contourArea(approximated_poly)) < 30000:
                x, y, w, h = cv2.boundingRect(approximated_poly)
                cells.append(np.array([x, y, w, h], dtype=np.int32))

    return np.array(cells, dtype=np.int32)


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

    edges = cv2.Canny(img, 30, 180)

    # findContours: boundaries of shapes having same intensity
    vertices = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(vertices)

    # sort contours based on the Area of a contour from big to small
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # find the actual board in the img
    board = None
    for contour in sorted_contours:
        approximated_poly = cv2.approxPolyDP(contour, 15, True)
        if len(approximated_poly) == 4:
            board = np.array(approximated_poly, dtype=np.int32)
            break

        elif math.fabs(cv2.contourArea(approximated_poly)) > 1000:
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
sort boxes from UtoB
"""


def sort_boxes(boxes):
    sorted_boxes = []
    U2B = np.argsort([box[1] for box in boxes])
    U2B_sorted = boxes[U2B]
    rows = np.vsplit(U2B_sorted, 9)
    for row in rows:
        L2R = np.argsort([box[0] for box in row])
        cols = row[L2R]
        for box in cols:
            sorted_boxes.append(box)

    return np.array(sorted_boxes)


"""
Projekt the board into a 900 * 900 window
"""


def get_perspektiveProjection(img, board, height=900, width=900):
    board = sort_points(board)
    board_vertecies = np.float32([board[0], board[1], board[2], board[3]])
    box_vertecies = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

    projektMatrix = cv2.getPerspectiveTransform(board_vertecies, box_vertecies)
    return cv2.warpPerspective(img, projektMatrix, (height, width))


def displayNumbers(img, numbers, color=(0, 255, 0)):
    """Displays 81 numbers in an image or mask at the same position of each cell of the board"""
    W = int(img.shape[1] / 9)
    H = int(img.shape[0] / 9)
    for i in range(9):
        for j in range(9):
            if numbers[(j * 9) + i] != 0:
                cv2.putText(img, str(numbers[(j * 9) + i]), (i * W + int(W / 2) - int((W / 4)), int((j + 0.7) * H)),
                            cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
    return img
