import glob
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
# import cv2 
import cv2 as cv
from matplotlib import pyplot as plt 
import os

import settings

from utils import PDFpage
from read_pdf import ReadPdf

DEBUG = settings.DEBUG

def show_wait_destroy(winname, img):
    if not DEBUG: return
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

def get_contour_precedence(contour, cols):
    print("contour, cols--", contour, cols)
    origin = cv.boundingRect(contour)
    print("origin--",origin, "-----",origin[1] * cols + origin[0])
    # return origin[1] * cols + origin[0]
    # return origin[1] * cols + origin[0]
    return cols + origin[0]


if __name__ == "__main__":

    INPUT_DIR = settings.INBOX

    pdfs = glob.glob(f'{INPUT_DIR}/*.pdf')
    for pdf in pdfs[:1]:
        print("pdf--",pdf)
        pdf_obj = ReadPdf(pdf)

        ############# To read page one  by one ##############
        for pg_num in range(1,pdf_obj.num_of_pages())[2:3]:

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

                # print(inverted_table)
                contours, hierarchy = cv.findContours(inverted_table, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                print("contours---",len(contours))

                # contours = sorted(contours, key=lambda ctr: cv.boundingRect(ctr)[0] + cv.boundingRect(ctr))
                # contours.sort(key=lambda x:get_contour_precedence(x, img.shape[1]))
                contours.sort(key=lambda x:get_contour_precedence(x, img.shape[1]))
                cv.drawContours(img, contours, -1, (0,255,0), -1)



                for i,cnt in enumerate(contours):
                    # print(cnt)
                    x,y,w,h = cv.boundingRect(cnt)
                    # compute the center of the contour
                    # M = cv.moments(cnt)
                    # cX = int(M["m10"] / M["m00"])
                    # cY = int(M["m01"] / M["m00"])

                    x = int(x + w / 2)
                    y = int(y + h / 2)
                    cv.putText(img, str(i), (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                cv.imwrite(f'{outbox}/pg-{pg_num}-img.png',img)
                show_wait_destroy('img',img)



            cv.destroyAllWindows()
        ############# To read page one  by one ##############


