import glob
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
# import cv2 
import cv2 as cv
from matplotlib import pyplot as plt 
import os
import math
import statistics 
from pytesseract import image_to_string
import pandas as pd


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
    # pdfs = glob.glob(f'{INPUT_DIR}/22123708.pdf')
    for pdf in pdfs[:1]:
        print("pdf--",pdf)
        pdf_obj = ReadPdf(pdf)

        ############# To read page one  by one ##############
        FRAMES = [] # one dataframe per page

        for pg_num in range(pdf_obj.num_of_pages())[2:4]:

            img,status = pdf_obj.read_page(pg_num)
            # if status:
            pdfPageObj = PDFpage(img)
            pdfPageObj.fix_page_orientation()

            img = pdfPageObj.get_image()  # cv image with fixed orientation

            
            gray,table,inverted_table,head,no_grid = pdfPageObj.get_grid()
            # pdfPageObj.get_grid()

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


            print(len(contours))
            cv.drawContours(img, contours, -1, (0,255,0), -1)
            for i,cnt in enumerate(contours):
                x,y,w,h = cv.boundingRect(cnt)
                # print(x,y,w,h)
                xc = int(x + w / 2)
                yc = int(y + h / 2)
                cv.putText(img, str(i), (xc,yc), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv.imwrite(f'{outbox}/pg-{pg_num}-detection.png',img)
            show_wait_destroy('img',img)

            # table cell bb
            table_cells_bbs = get_row_list(contours) # tuple of bounding boxes

            no_of_data_cols = statistics.mode([len(row) for row in table_cells_bbs])
            print('no_of_data_cols--',no_of_data_cols)

            table_cells_imgs = pdfPageObj.table_cell_list(table_cells_bbs,img)  # tuple of cell images
            table_cells_imgs = filter(lambda x: len(x) == no_of_data_cols,table_cells_imgs)


            # data = (('A','B','C','D'),(1,2,3,4),(5,6,7,8),(9,10,11,12))
            df = pd.DataFrame(tuple(table_cells_imgs)[:5])

            df = df.applymap(image_to_string)
            df.columns = df.iloc[0]
            df.drop(df.index[0],inplace=True)

            FRAMES.append(df)

        print(FRAMES)
        result_df = pd.concat(FRAMES,axis=0,ignore_index=True)

        print(result_df)

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(f'{outbox}.xlsx', engine='xlsxwriter')

        # Convert the dataframe to an XlsxWriter Excel object.
        result_df.to_excel(writer, sheet_name='test',header=True, index=False)

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

                # print(df.columns)

                # for table_cells_img in table_cells_imgs:
                #     print(len(table_cells_img))
                    # IMG = np.hstack(([img for img in table_cells_img]))
                    # show_wait_destroy(f'cell_img',cell_img)


                # for cell_img_row in table_cells_imgs:

                #     print("=================--*texts--==============")
                #     map()
                #     texts = map(image_to_string,cell_img_row)
                #     print("--*texts--", *texts)


                # #------------------ for demo ------------------#
                # for cell_img_list in list(table_cells_imgs)[-3:-2]:
                #     print(len(cell_img_list))
                #     for cell_img in cell_img_list:
                #         print("------")
                #         cell_img[cell_img > 200] = 255
                #         print(image_to_string(cell_img))
                #         show_wait_destroy(f'cell_img',cell_img)
                # #------------------- for demo ------------------#



                # print(len(row_list))

                # ------------------------------------------------------- #
                # Under testing
                # contours.sort(key=lambda x:ordering(x, img.shape[0]))
                # row_range_list = row_range_list(contours)
                # table = dict.fromkeys(row_range_list,[])

                # for tup_range,value_list in table.items():
                #     table[tup_range] = list(filter(lambda cnt: tup_range[0] <  math.ceil(( cnt[0][0][1]+ cnt[1][0][1])/2)  < tup_range[1], contours ))
                # -------------------------------------------------------- #



                # print(len(contours))
                # cv.drawContours(img, contours, -1, (0,255,0), -1)
                # for i,cnt in enumerate(contours):
                #     x,y,w,h = cv.boundingRect(cnt)
                #     # print(x,y,w,h)
                #     xc = int(x + w / 2)
                #     yc = int(y + h / 2)
                #     cv.putText(img, str(i), (xc,yc), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # cv.imwrite(f'{outbox}/pg-{pg_num}-detection.png',img)
                # show_wait_destroy('img',img)



            # cv.destroyAllWindows()
        ############# To read page one  by one ##############


