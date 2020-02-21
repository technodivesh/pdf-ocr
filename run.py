#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__      = "Divesh Chandolia"


import glob
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt 
import os
import math
import statistics 
import itertools
from pytesseract import image_to_string
import pandas as pd
from get_ba_check_details import Cheque


import settings
import config

from utils import PDFpage
from read_pdf import ReadPdf
from pprint import pprint

DEBUG = settings.DEBUG
COLS = config.PROFESSIONAL_REMITTANCE_ADVICE.keys()

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
    
    tolerance_factor = 15
    origin = cv.boundingRect(contour)
    # print( f"(( {origin[1]} // {tolerance_factor}) * {tolerance_factor}) * {cols} + {origin[0]}" )
    # print(((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0])
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

def get_bb_precedence(bb,cols):

    tolerance_factor = 15
    return ((bb[1] // tolerance_factor) * tolerance_factor) * cols + bb[0]



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

def get_bb_array(contours):
    
    # (x,y,w,h )
    return ( cv.boundingRect(cnt) for cnt in contours )

def image_to_string_demo(img):
    return "divesh"



if __name__ == "__main__":

    INPUT_DIR = settings.INBOX

    # pdfs = glob.glob(f'{INPUT_DIR}/*229.pdf')
    pdfs = glob.glob(f'{INPUT_DIR}/*22123708.pdf')
    # pdfs = glob.glob(f'{INPUT_DIR}/*731.pdf')
    

    for pdf in pdfs[:1]:
        print("pdf--",pdf)
        pdf_obj = ReadPdf(pdf)

        COLUMNS = pd.DataFrame()
        result_df = pd.DataFrame()


        # To write in file
        pdfNameOnly = os.path.splitext(os.path.basename(pdf))[0]
        outbox = os.path.join(settings.OUTBOX,pdfNameOnly)
        if not os.path.exists(outbox):
            os.makedirs(outbox)

        # #------ Cheque Info -----------#
        # cheque_dict ={}
        # check_page_img,status = pdf_obj.read_page(1)
        # # show_wait_destroy('check_page_img', check_page_img)
        # cheque = Cheque(check_page_img)
        # cheque.fix_page_orientation()
        # cheque.find_cheque_details()

        # # print('--------------------')
        # # print(cheque.amount)
        # # print(cheque.number)
        # # print(cheque.date)
        # # print(cheque.address)

        # data = [[cheque.amount,cheque.number,cheque.date,cheque.address]]
        # cheque_df = pd.DataFrame(data, columns = config.PROFESSIONAL_REMITTANCE_ADVICE_CHECK) 
        # # cheque_df = pd.DataFrame(data, columns = ['Amount', 'Number','Date','Address']) 
        # cheque_dict = {key:val[0] for key,val in cheque_df.to_dict().items()}
        # print(cheque_df)

        # #------ Cheque Info ---------#

        # exit()
        Meta = False

        ############# To read page one  by one ##############
        print("pdf_obj.num_of_pages()---", pdf_obj.num_of_pages())
        for pg_num in range(pdf_obj.num_of_pages())[2:]:

            print("------------pg_num-----", pg_num)
            img,status = pdf_obj.read_page(pg_num)
            # if status:
            pdfPageObj = PDFpage(img)
            pdfPageObj.fix_page_orientation()

            img = pdfPageObj.get_image()  # cv image with fixed orientation
            show_wait_destroy('img-cv',img)
            meta_dict = {}

            
            img,gray,table,inverted_table,head,no_grid = pdfPageObj.get_grid()
            # img = gray
            # DENOISE IT img

            # show_wait_destroy('gray',gray)
            # show_wait_destroy(f'table-{table.shape}',table)
            # show_wait_destroy('head',head)
            # show_wait_destroy('no_grid',no_grid)
            # show_wait_destroy('inverted_table',inverted_table)

            # # ----------- Page Head Info ----------#
            # if not Meta:
            #     contours, hierarchy = cv.findContours(head, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            #     contours = list(filter(lambda x: len(x) > 2, contours))
            #     contours.sort(key=lambda x:get_contour_precedence(x, img.shape[1]))

            #     print(len(contours))
            #     cv.drawContours(img, contours, -1, (0,255,0), -1)
            #     for i,cnt in enumerate(contours):
            #         x,y,w,h = cv.boundingRect(cnt)
            #         # print(x,y,w,h)
            #         xc = int(x + w / 2)
            #         yc = int(y + h / 2)
            #         cv.putText(img, str(i), (xc,yc), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
            #     cv.imwrite(f'{outbox}/pg-{pg_num}-head-detection.png',img)
            #     show_wait_destroy('head-img',img)

            #     table_cells_bbs = get_row_list(contours) # tuple of bounding boxes
            #     no_of_data_cols = statistics.mode([len(row) for row in table_cells_bbs])

            #     table_cells_imgs = pdfPageObj.table_cell_list(table_cells_bbs,img)  # tuple of cell images
            #     table_cells_imgs = filter(lambda x: len(x) == no_of_data_cols,table_cells_imgs)

            #     meta_df = pd.DataFrame(tuple(table_cells_imgs))
            #     meta_df = meta_df.applymap(image_to_string)
            #     # convert Col to header
            #     # meta_df = pd.DataFrame([list(meta_df[1])], columns = list(meta_df[0])) 
            #     meta_df = pd.DataFrame([list(meta_df[1])], columns = list(meta_df[0])) 
            #     print(meta_df)
            #     _meta_df = meta_df.copy()
            #     # _meta_df.columns = config.PROFESSIONAL_REMITTANCE_ADVICE_META
            #     print(_meta_df)
            #     meta_dict = {key:val[0] for key,val in _meta_df.to_dict().items()}

            #     Meta = True

            # # ----------- Page Head Info ----------#
            # exit()


            # cv.imwrite(f'{outbox}/pg-{pg_num}-first.png',img)
            cv.imwrite(f'{outbox}/pg-{pg_num}-gray.png',gray)
            # cv.imwrite(f'{outbox}/pg-{pg_num}-table.png',table)
            # cv.imwrite(f'{outbox}/pg-{pg_num}-head.png',head)
            # cv.imwrite(f'{outbox}/pg-{pg_num}-no_grid.png',no_grid)
            # cv.imwrite(f'{outbox}/pg-{pg_num}-inverted_table.png',inverted_table)

            contours, hierarchy = cv.findContours(inverted_table, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours = list(filter(lambda x: len(x) > 3, contours))

            # contours = (filter( ,contours))
            # contours.sort(key=lambda x:get_contour_precedence(x, img.shape[1]))

            print(len(contours))
            # cv.drawContours(img, contours, -1, (0,255,0), -1)
            
            bb_array = get_bb_array(contours)
            bb_array = list(bb_array)
            bb_array.sort(key=lambda x:get_bb_precedence(x, img.shape[1]))
            # print(list(bb_array))

            for i, (x,y,w,h) in enumerate(list(bb_array)):
                # print(x,y,w,h)
                start_point = (x, y) 
                end_point = (x+w, y+h-1) 
                color = (0, 255, 0) 
                thickness = -1
                cv.rectangle(img, start_point, end_point, color, thickness) 
                xc = int(x + w / 2)
                yc = int(y + h / 2)
                cv.putText(img, str(i), (xc,yc), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv.imwrite(f'{outbox}/pg-{pg_num}-bb-table.png',img)
            # show_wait_destroy('img',img)
            # exit()

            for i,cnt in enumerate(contours):
                x,y,w,h = cv.boundingRect(cnt)
                # -------------------------------------------# 
                start_point = (x, y) 
                end_point = (x+w, y+h-2) 
                color = (255, 0, 0) 
                thickness = -1
                cv.rectangle(img, start_point, end_point, color, thickness) 
                # -------------------------------------------# 
                # print(x,y,w,h)
                xc = int(x + w / 2)
                yc = int(y + h / 2)
                cv.putText(img, str(i), (xc,yc), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv.imwrite(f'{outbox}/pg-{pg_num}-detection.png',img)
            show_wait_destroy('img',img)
            # exit()

            # table cell bb
            table_cells_bbs = get_row_list(contours) # tuple of bounding boxes

            no_of_data_cols = statistics.mode([len(row) for row in table_cells_bbs])
            print('no_of_data_cols--',no_of_data_cols)

            table_cells_imgs = pdfPageObj.table_cell_list(table_cells_bbs,img)  # tuple of cell images
            table_cells_imgs = filter(lambda x: len(x) == no_of_data_cols,table_cells_imgs)


            # #------------------ for testing only ------------------#
            # for cell_img_list in list(table_cells_imgs)[14:]:
            #     print(len(cell_img_list))
            #     show_wait_destroy("np.hstack--",np.hstack([img for img in cell_img_list]))
            #     for cell_img in cell_img_list:
            #         print("------")
            #         cell_img[cell_img > 200] = 255
            #         ###############################
            #         ###############################
            #         ###############################
            #         print(image_to_string(cell_img))
            #         ###############################
            #         ###############################
            #         ###############################
            #         show_wait_destroy(f'cell_img',cell_img)
            # cv.destroyAllWindows()
            # exit()
            # #------------------- for testing only ------------------#

            # Create DataFrame
            df = pd.DataFrame(tuple(table_cells_imgs))

            # Apply function to all cells
            ###############################
            ###############################
            ###############################
            df = df.applymap(image_to_string_demo)
            ###############################
            ###############################
            ###############################

            # df.columns = df.iloc[0]

            if COLUMNS.empty:
                COLUMNS = df.iloc[0]
            df.drop(df.index[0],inplace=True)

            print("--------appending dataframe------------")
            result_df = result_df.append(df,ignore_index=True)

        result_df.columns = COLUMNS


        
        print("-------- Saving XLS file Start -----------")
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(f'{outbox}/{pdfNameOnly}.xlsx', engine='xlsxwriter')

        # Convert the dataframe to an XlsxWriter Excel object.
        result_df.to_excel(writer, sheet_name='PCP-Information',header=True, index=False)
        cheque_df.to_excel(writer, sheet_name='Cheque',header=True, index=False)
        meta_df.to_excel(writer, sheet_name='Meta',header=True, index=False)

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()
        print("-------- Saving XLS file Done -----------")


    
        print("-------- Formating for CSV -----------")
        # Formating
        # Update col names 
        result_df.columns = COLS

        # Replace blank with NaN
        result_df.replace('',np.nan,inplace=True)

        # Delete blank rows
        result_df.dropna(axis=0, how='all', thresh=None, subset=None, inplace=True)

        # Fill last column with values - forward fill
        result_df[result_df.columns[-1]].fillna(method='bfill',inplace = True)

        # group by last columns value
        dfs = result_df.groupby(result_df.columns[-1],sort=False)

        # remove single liners from list of df
        dfs = list(_df.drop(_df.index[-1]) for i,_df in dfs if len(_df)>1)

        ##################################################################
        # getting Carreir Name 
        for df in dfs:
            DROP = []
            CARRIER = ''
            
            for i,row in enumerate(list(df.iterrows())): 
                if df.iloc[i][1:-1].isna().all():
                    
                    CARRIER = df.iloc[i][0] 
                    DROP.append(i)

            if DROP:
                df.drop(df.index[DROP[0]],inplace=True)
        ##################################################################


        # print(dfs)
        ALL_COLS = list(config.PROFESSIONAL_REMITTANCE_ADVICE.values()) + [(config.PROFESSIONAL_REMITTANCE_ADVICE_CHECK)] + [(config.PROFESSIONAL_REMITTANCE_ADVICE_META)]
        ALL_COLS = list(itertools.chain.from_iterable(ALL_COLS))
        
        frames = []
        for df in dfs:

            new_df = pd.DataFrame(columns=ALL_COLS)
            
            temp_dict = {}

            for cb_col in df.columns: # cb is conbined column
        #         print(df[cols])
                if cb_col in ('LASTNAME__PATIENT_ACCOUNT','FIRSTNAME__MEMBER_ID','CLAIM_NUMBER__RECVDDT__SERVPROV'):
        #             print(df[cb_col].dropna().tolist())
        #             print(config.PROFESSIONAL_REMITTANCE_ADVICE[cb_col])
                    temp_dict = {**temp_dict,**dict(list(zip(config.PROFESSIONAL_REMITTANCE_ADVICE[cb_col],df[cb_col].dropna().tolist())))}
            
        #     print(df['PROCEDURE__MODIFIER'].tolist())
        #     print(df.columns[3:])
            for i in df.index:
                if df.loc[i][['PROCEDURE__MODIFIER']].isnull().values.any():continue
        #             print(df.loc[i]['PROCEDURE__MODIFIER':].tolist())
                for c in df.columns[3:-1]:
                    if c == 'DATE_OF_SERVICE__FROM_THRU':
                        print(df.loc[i][c])
                        try:
                            F,T = df.loc[i][c].split('-') 
                        # except Exception as e:
                        #     print('e--1',e)
                        #     F,T = df.loc[i][c].split('~') 
                        # except Exception as e:
                        #     print('e--2',e)
                        #     F,T = df.loc[i][c].split(' ')
                        except Exception as e:
                            print('e--3',e)
                            F,T = ['121212','121212']

                        F =  str(F) + str(T[-2:])
                                    
                        F = ''.join([n for n in F if n.isdigit()])
                        T = ''.join([n for n in T if n.isdigit()])
                        
                        try:
                            Fd = datetime.datetime.strptime(F[-6:], '%m%d%y').strftime('%m/%d/%y')
                        except:
                            Fd = F
                        try:
                            Td = datetime.datetime.strptime(T[-6:], '%m%d%y').strftime('%m/%d/%y')
                        except:
                            Td = T
                        print('F--------',F,Fd)
                        print('T--------',T,Td)
                        FT = [Fd,Td]
                        temp_dict = {**temp_dict,**dict(list(zip(config.PROFESSIONAL_REMITTANCE_ADVICE[c],FT)))}
                    elif c == 'PROCEDURE__MODIFIER':
                        PM = [df.loc[i][c][:5],df.loc[i][c][5:]]
                        temp_dict = {**temp_dict,**dict(list(zip(config.PROFESSIONAL_REMITTANCE_ADVICE[c],PM)))}
                        
                    else:
                        DATA = [df.loc[i][c]]
                        temp_dict = {**temp_dict,**dict(list(zip(config.PROFESSIONAL_REMITTANCE_ADVICE[c],DATA)))}
                        

#               pprint(temp_dict)
                temp_dict = {**temp_dict,**cheque_dict}
                temp_dict = {**temp_dict,**meta_dict}            
                new_df = new_df.append(temp_dict,ignore_index=True)

                    


            #     print(new_df)
            frames.append(new_df)
            
        result = pd.concat(frames)

        result.to_csv(f'{outbox}/{pdfNameOnly}.csv', index=False)
        print(result)