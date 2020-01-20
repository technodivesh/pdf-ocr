"""
@file morph_lines_detection.py
@brief Use morphology transformations for extracting horizontal and vertical lines sample code
"""
import numpy as np
import sys
import cv2 as cv
import os


import glob
from pdf2image import convert_from_path
from PIL import Image
# from matplotlib import pyplot as plt 


from utils import PDFpage
from read_pdf import ReadPdf
import settings

DEBUG = True

def show_wait_destroy(winname, img):
    if not DEBUG: return
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

def closed_grid(grid):

    contours, hierarchy = cv.findContours(grid, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    print(type(contours), len(contours))

    largest_c = max(contours, key = cv.contourArea)
    x,y,w,h = cv.boundingRect(largest_c)
    cv.rectangle(grid,(x,y),(x+w,y+h),255,1)

    # show_wait_destroy('closed_grid',grid)
    return grid,x,y,w,h


def complete_grid(grid,img,*xywh):

    "It will add only horizontal lines"

    x,y,w,h = (xywh)

    # # if not on gray scale
    if len(img.shape) > 2:
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # print(grid[y+1,x+1])     # row, col
    temp = cv.add(grid,img)   # temp is the binary image

    ####################### creating horizontal lines -start ###################
    height,width = grid.shape[:2]
    mask = np.zeros((height,width), np.uint8)

    white_list = []
    black_list = []
    wt=[]
    bl=[]
    for row in range(y,y+h):

        if all(temp[row,x:x+w] > 127): # blank / white line ; No character
            wt.append(row)

            if bl:
                black_list.append(tuple(bl))
                bl = []
            # mask[row,x:x+w] = 255
        else: # character occures, make / keep black
            bl.append(row)
            if wt:
                white_list.append(tuple(wt))
                wt = []


    for tup in white_list[1:-1]:
        # print(tup)
        if len(tup) >= 3:
            row = tup[len(tup) // 2]
            mask[row,x:x+w] = 255

    grid = cv.add(grid,mask)
    return grid

def main(src):

    # cv.imshow("src", src)

    # Transform source image to gray if it is not already
    if len(src.shape) != 2:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        gray = src

    org_gray = gray.copy()
    # Show gray image
    # show_wait_destroy("gray", gray)
    
    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                cv.THRESH_BINARY, 15, -2)
    # Show binary image
    # show_wait_destroy("binary", bw)
    # [bin]
    # [init]
    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(bw)
    vertical = np.copy(bw)

    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 10
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)
    # Show extracted horizontal lines
    # show_wait_destroy("horizontal", horizontal)

    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = rows // 10
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv.erode(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure)
    # Show extracted vertical lines
    # show_wait_destroy("vertical", vertical)

    # Inverse vertical image

    merge = cv.add(vertical ,horizontal)




    rows_to_be_black = int(vertical.shape[0] // 4.5)

    table = merge.copy()
    head = merge.copy()

    table[:rows_to_be_black, :] = 0
    table[:, :5] = 0
    head[rows_to_be_black:, :] = 0
    head[:5, :] = 0

    # Step 1
    edges = cv.adaptiveThreshold(table, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, -2)
    # show_wait_destroy("edges", edges)
    # Step 2
    kernel = np.ones((2, 2), np.uint8)
    edges = cv.dilate(edges, kernel)
    # show_wait_destroy("dilate", edges)
    table = edges


    table,*xywh = closed_grid(table)
    no_grid = cv.add(org_gray,table)
    table = complete_grid(table,no_grid,*xywh)




    # show_wait_destroy("table", table)

    return org_gray,table,head, no_grid


if __name__ == "__main__":

    INPUT_DIR = settings.INBOX

    pdfs = glob.glob(f'{INPUT_DIR}/*.pdf')
    for pdf in pdfs[:3]:  # To read all files delete [:3]

        pdf_obj = ReadPdf(pdf)

        for pg_num in range(1,pdf_obj.num_of_pages())[2:3]:    # to read all pages remove [2:3]

            page = pdf_obj.read_page(pg_num)
            if page:
            #     print(page)
                img = np.array(page.convert('RGB')) # BGR image in opencv
                print(img.shape)

                pdfPageObj = PDFpage(img)
                pdfPageObj.fix_page_orientation()

                img = pdfPageObj.get_image()
                gray,table,head,no_grid = main(img)

                # show_wait_destroy('gray',gray)
                # show_wait_destroy('table',table)
                # show_wait_destroy('head',head)
                # show_wait_destroy('no_grid',no_grid)

                cv.destroyAllWindows()


                # To write in file
                pdfNameOnly = os.path.splitext(os.path.basename(pdf))[0]
                outbox = os.path.join(settings.OUTBOX,pdfNameOnly)
                if not os.path.exists(outbox):
                    os.makedirs(outbox)

                cv.imwrite(f'{outbox}/pg-{pg_num}-gray.png',gray)
                cv.imwrite(f'{outbox}/pg-{pg_num}-table.png',table)
                cv.imwrite(f'{outbox}/pg-{pg_num}-head.png',head)
                cv.imwrite(f'{outbox}/pg-{pg_num}-no_grid.png',no_grid)

