#Droidcam implementation by https://github.com/cardboardcode/droidcam_simple_setup
#from _cffi_backend import callback

from OpenCVHelpers import *
from TesorflowHelpers import *
from sudoku import Sudoku
from VideoStreamer import *

from multiprocessing.dummy import Pool

HTTP = 'http://'
IP_ADDRESS = '192.168.178.21'
URL =  HTTP + IP_ADDRESS + ':4747/mjpegfeed?640x480'

IMG_SIZE = 48

def find_cells(perspective_img):
    cells = locate_cells(perspective_img)
    if len(cells) == 81:
        cells = sort_boxes(cells)
        box_imgs = get_box_imgs(perspective_img, cells, IMG_SIZE)
        for cell in cells:
            cv2.rectangle(perspective_img, (cell[0], cell[1]), (cell[0] + cell[2], cell[1] + cell[3]), (36, 255, 12), 3)
        return perspective_img, box_imgs
    else:
        pass
        #print("couldn't find the cells in sudoku! make sure sudoku is 9x9")

def find_sudoku(img):
    processed_img = preprocess_img(img)
    board = locate_sudoku(processed_img)
    if board is not None:
        perspective_img = get_perspektiveProjection(img, board)
        perspective_imgs = []

        perspective_imgs.append(perspective_img)
        for i in range(1, 4):
            perspective_imgs.append(cv2.rotate(perspective_imgs[i-1], cv2.ROTATE_90_CLOCKWISE))
        return perspective_imgs
    else:
        pass
        #print("couldn't locate the board! please move the camera around")

def solve(cells):
    if cells is not None:
        cell, boximages = cells
        show_image(cell, "detected cells")
        board_numbers = get_prediction(boximages)
        if board_numbers.shape == (9, 9):

            predicted_numbers = board_numbers.flatten()
            puzzle = Sudoku(3, 3, board=board_numbers.tolist())
            puzzle.show_full()

            solved_board_nums = np.array(puzzle.solve(raising=True).board)

            # create a binary array of the predicted numbers. 0 means unsolved numbers of sudoku and 1 means given number.
            binArr = np.where(np.array(predicted_numbers)>0, 0, 1)

            # get only solved numbers for the solved board
            flat_solved_board_nums = solved_board_nums.flatten()*binArr

            # create a mask
            mask = np.zeros_like(cell)
            # displays solved numbers in the mask in the same position where board numbers are empty
            solved_board_mask = displayNumbers(mask, flat_solved_board_nums)

            combined = cv2.addWeighted(cell, 0.7, solved_board_mask, 1, 0)
            cv2.destroyWindow("detected cells")
            cv2.imshow('frame', combined)
            return True
    return False

def camdroid():
    print("[ droidcam.py ] - Initializing...")

    # Opening video stream of ip camera via its url
    #cap = cv2.VideoCapture(URL)
    cap = VideoCaptureHelper(URL)
    # Corrective actions printed in the even of failed connection.
    if cap.cap.isOpened() is not True:
        print ('Not opened.')
        print ('Please ensure the following:')
        print ('1. DroidCam is not running in your browser.')
        print ('2. The IP address given is correct.')

    BOARD_SOLVED = False

    # Connection successful. Proceeding to display video stream.
    while True:
        if BOARD_SOLVED:
            print("scan another board?")
            if cv2.waitKey(0) == ord('y'):
                BOARD_SOLVED = False
                continue
            else:
                break

        # Capture frame-by-frame
        frame = cap.read()
        if frame is not None:
            show_image(frame, 'frame')
            board = find_sudoku(frame)
            if board is not None:
                for imgs in board:
                    cells = find_cells(imgs)
                    if cells is not None:
                        if not BOARD_SOLVED:
                            try:
                                BOARD_SOLVED = solve(cells)
                                break
                            except:
                                continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.cap.release()
    cv2.destroyAllWindows()


def main():
    camdroid()

if __name__ == '__main__':
    main()
