import glob
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
# import cv2 
import cv2 as cv
from matplotlib import pyplot as plt 
import os
import math

import settings

from utils import PDFpage
from read_pdf import ReadPdf
from pprint import pprint

DEBUG = settings.DEBUG

def show_wait_destroy(winname, img):
    if not DEBUG: return
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)


def ordering(contour, rows):
    origin = cv.boundingRect(contour)
    return origin[0] * rows # + origin[0]

def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

def row_range_list(contours):

    range_list = []
    for i, cnt in enumerate(contours):
        x,y,w,h = cv.boundingRect(cnt)
        if i == 0:
            range_list.append((y,y+h))
        elif y < range_list[-1][0]:
            range_list.append((y,y+h))
        else:break
    
    return range_list

def get_row_list(contours):

    row_list = []
    col_list = []
    for cnt in contours:
        x,y,w,h = cv.boundingRect(cnt)
        if not col_list:
            col_list.append((x,y,w,h))
        elif x > col_list[-1][0]:
            col_list.append((x,y,w,h))
        else:
            row_list.append(tuple(col_list))
            col_list = []
            col_list.append((x,y,w,h))

    row_list.append(tuple(col_list)) 

    return tuple(row_list)





if __name__ == "__main__":

    INPUT_DIR = settings.INBOX

    pdfs = glob.glob(f'{INPUT_DIR}/*229.pdf')
    for pdf in pdfs[:1]:
        print("pdf--",pdf)
        pdf_obj = ReadPdf(pdf)

        ############# To read page one  by one ##############
        for pg_num in range(pdf_obj.num_of_pages())[2:3]:

            img,status = pdf_obj.read_page(pg_num)
            if status:
                pdfPageObj = PDFpage(img)
                pdfPageObj.fix_page_orientation()

                img = pdfPageObj.get_image()  # cv image with fixed orientation

                gray,table,inverted_table,head,no_grid = pdfPageObj.get_grid(img)

                # show_wait_destroy('gray',gray)
                # show_wait_destroy('table',table)
                # show_wait_destroy('head',head)
                # show_wait_destroy('no_grid',no_grid)
                show_wait_destroy('inverted_table',inverted_table)

                # To write in file
                pdfNameOnly = os.path.splitext(os.path.basename(pdf))[0]
                outbox = os.path.join(settings.OUTBOX,pdfNameOnly)
                if not os.path.exists(outbox):
                    os.makedirs(outbox)

                # cv.imwrite(f'{outbox}/pg-{pg_num}-first.png',img)
                # cv.imwrite(f'{outbox}/pg-{pg_num}-gray.png',gray)
                # cv.imwrite(f'{outbox}/pg-{pg_num}-table.png',table)
                # cv.imwrite(f'{outbox}/pg-{pg_num}-head.png',head)
                # cv.imwrite(f'{outbox}/pg-{pg_num}-no_grid.png',no_grid)
                # cv.imwrite(f'{outbox}/pg-{pg_num}-inverted_table.png',inverted_table)

                contours, hierarchy = cv.findContours(inverted_table, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                contours = list(filter(lambda x: len(x) > 2, contours))
                contours.sort(key=lambda x:get_contour_precedence(x, img.shape[1]))

                row_list = get_row_list(contours)
                for row in row_list:
                    print(f"{row}--{len(row)}")
                    for x,y,w,h in row:
                        print( x,y,w,h)
                        # show_wait_destroy(f'test',img[y:y+h,x:x+w])

                print(len(row_list))

                # ------------------------------------------------------- #
                # Under testing
                # contours.sort(key=lambda x:ordering(x, img.shape[0]))
                # row_range_list = row_range_list(contours)
                # table = dict.fromkeys(row_range_list,[])

                # for tup_range,value_list in table.items():
                #     table[tup_range] = list(filter(lambda cnt: tup_range[0] <  math.ceil(( cnt[0][0][1]+ cnt[1][0][1])/2)  < tup_range[1], contours ))
                # -------------------------------------------------------- #
                
                # pprint(table)
                # for k,v in table.items():
                #     print(k,len(v))




                print(len(contours))
                cv.drawContours(img, contours, -1, (0,255,0), -1)
                for i,cnt in enumerate(contours):
                    # print("---------------------------------",len(cnt),i)
                    # print(cnt[0])
                    # print(cnt[1])
                    # print(cnt[2])
                    # print(cnt[3])
                    x,y,w,h = cv.boundingRect(cnt)
                    # print(x,y,w,h)
                    xc = int(x + w / 2)
                    yc = int(y + h / 2)
                    cv.putText(img, str(i), (xc,yc), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                cv.imwrite(f'{outbox}/pg-{pg_num}-detection.png',img)
                show_wait_destroy('img',img)



            cv.destroyAllWindows()
        ############# To read page one  by one ##############


