#Droidcam implementation by https://github.com/cardboardcode/droidcam_simple_setup
#from _cffi_backend import callback

from OpenCVHelpers import *
from SudokuSolverHelper import solveSudoku
from TesorflowHelpers import *

from multiprocessing.dummy import Pool

HTTP = 'http://'
IP_ADDRESS = '192.168.178.21'
URL =  HTTP + IP_ADDRESS + ':4747/mjpegfeed?640x480'

IMG_SIZE = 48



def cellPatching(cells):
    print(cells)

    # Find potential cells which where missed in grid
    # A cell is the right top point of a box and has width and height additionally
    # A potential cell is adjacent to at least 4 other cells around
    # But with the points you can just look for top, bottom, left and right for another point then connect
    # and make box from point distances and found point remaining value
    print("cells")
    print(cells)
    return cells

s = False

def find_sudoku(img):
    processed_img = preprocess_img(img)
    #show_image(processed_img)
    board = locate_sudoku(processed_img)
    if board is not None:
        perspective_img = get_perspektiveProjection(img, board)
        #show_image(perspective_img)
        cells = locate_cells(perspective_img)
        if len(cells) == 81:
            cells = sort_boxes(cells)
            """box_imgs = get_box_imgs(perspective_img, cells, IMG_SIZE)
            board_numbers = get_prediction(box_imgs)
            print(board_numbers)
            if board_numbers.shape == (9, 9):
                #solveSudoku(board_numbers)
                if not globals()["s"]:
                    print("\nsolved grid: ")
                    globals()["s"] = True
                    #pool = Pool(processes=1)
                    print("solving...")
                    #pool.apply_async(solveSudoku, board_numbers, callback=globals()["s"])
                    globals()["s"] = solveSudoku(board_numbers)
                else:
                    print(globals()["s"])"""
            for cell in cells:
                cv2.rectangle(perspective_img, (cell[0], cell[1]), (cell[2], cell[3]), (36, 255, 12), 3)
            return perspective_img
        else:
            pass
            #print("couldn't find the cells in sudoku! make sure sudoku is 9x9")
    else:
        pass
        #print("couldn't locate the board! please move the camera around")

def camdroid():
    print("[ droidcam.py ] - Initializing...")

    # Opening video stream of ip camera via its url
    cap = cv2.VideoCapture(URL)

    # Corrective actions printed in the even of failed connection.
    if cap.isOpened() is not True:
        print ('Not opened.')
        print ('Please ensure the following:')
        print ('1. DroidCam is not running in your browser.')
        print ('2. The IP address given is correct.')

    # Connection successful. Proceeding to display video stream.
    while cap.isOpened() is True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        img = find_sudoku(frame)
        if img is not None:
            cv2.imshow('frame', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    camdroid()

if __name__ == '__main__':
    main()
