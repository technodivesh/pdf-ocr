#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__      = "Divesh Chandolia"

import pytesseract
import cv2 as cv
import imutils
import numpy as np

DEBUG = False

def show_wait_destroy(winname, img):
    if not DEBUG: return
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)


class PDFpage():

    def __init__(self, img):

        img[img[:,:,:]>200] = 255
        self.img = img
        self.org_img = img.copy()
        self.gray = '' 

    def fix_page_orientation(self):

        print("111111111111111",self.img.shape)
        scale_percent = 200 # percent of original size
        width = int(self.img.shape[1] * scale_percent / 100)
        height = int(self.img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv.resize(self.img, dim, interpolation = cv.INTER_AREA)

        print("2222222222222",resized.shape)

        #############################################################
        # rotate the image to deskew it
        angle = -4
        (h, w) = resized.shape[:2]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        resized_tilt = cv.warpAffine(resized, M, (w, h),
            flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
        # self.set_image()
        show_wait_destroy("resized_tilt--",resized_tilt)
        ##############################################################

        resized_tilt_gray = cv.cvtColor(resized_tilt, cv.COLOR_BGR2GRAY)
        show_wait_destroy("resized_tilt_gray--",resized_tilt_gray)
        resized_tilt_gray_inv = cv.bitwise_not(resized_tilt_gray)
        
        # threshold the image, setting all foreground pixels to
        # 255 and all background pixels to 0
        resized_tilt_gray_inv_thresh = cv.threshold(resized_tilt_gray_inv, 0, 255,
            cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

        
        # grab the (x, y) coordinates of all pixel values that
        # are greater than zero, then use these coordinates to
        # compute a rotated bounding box that contains all
        # coordinates
        coords = np.column_stack(np.where(resized_tilt_gray_inv_thresh > 0))
        angle = cv.minAreaRect(coords)[-1]
        
        # the `cv2.minAreaRect` function returns values in the
        # range [-90, 0); as the rectangle rotates clockwise the
        # returned angle trends to 0 -- in this special case we
        # need to add 90 degrees to the angle
        if angle < -45:
            angle = -(90 + angle)
        
        # otherwise, just take the inverse of the angle to make
        # it positive
        else:
            angle = -angle

        print(f"angle -- {angle}")

        # rotate the image to deskew it
        (h, w) = resized_tilt.shape[:2]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        resized = cv.warpAffine(resized_tilt, M, (w, h),
            flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

        scale_percent = 50 # percent of original size
        width = int(resized.shape[1] * scale_percent / 100)
        height = int(resized.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized_back = cv.resize(resized, dim, interpolation = cv.INTER_AREA)

        print("33333333",resized_back.shape)
        self.set_image(resized_back)


    def get_image(self):
        return self.img

    def set_image(self,img):
        self.img = img
        self.gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    def closed_grid(self,grid):
        "It enclosed the grid in a rectangular box"

        contours, hierarchy = cv.findContours(grid, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        print(type(contours), len(contours))

        largest_c = max(contours, key = cv.contourArea)
        x,y,w,h = cv.boundingRect(largest_c)
        cv.rectangle(grid,(x,y),(x+w,y+h),255,5)
        show_wait_destroy('closed_grid',grid)
        return grid,x,y,w,h


    def complete_grid(self,grid,img,*xywh):

        "It adds only horizontal lines"
        x,y,w,h = (xywh)

        # if not on gray scale
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


    def get_grid(self):
        " It returns org_gray,table,head,no_grid"

        gray = self.gray  
        org_gray = gray.copy()
        # Show gray image
        show_wait_destroy("gray_gray", gray)
        
        # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
        gray = cv.bitwise_not(gray)
        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                    cv.THRESH_BINARY, 15, -2)
        # Show binary image

        # testing
        kernel = np.ones((2,2), np.uint8) 
        thresh = cv.dilate(thresh, kernel)

        show_wait_destroy("thresh", thresh)
        # print(thresh)
        # [bin]
        # [init]
        # Create the images that will use to extract the horizontal and vertical lines
        horizontal = np.copy(thresh)
        vertical = np.copy(thresh)

        # Specify size on horizontal axis
        cols = horizontal.shape[1]
        horizontal_size = cols // 10
        # Create structure element for extracting horizontal lines through morphology operations
        horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 2))
        # Apply morphology operations
        show_wait_destroy("horizontalStructure", horizontalStructure)
        horizontal = cv.erode(horizontal, horizontalStructure)
        show_wait_destroy("horizontal", horizontal)
        horizontal = cv.dilate(horizontal, horizontalStructure)
        # Show extracted horizontal lines

        # Specify size on vertical axis
        rows = vertical.shape[0]
        verticalsize = rows // 10
        # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (2, verticalsize))
        show_wait_destroy("verticalStructure", verticalStructure)
        # Apply morphology operations
        vertical = cv.erode(vertical, verticalStructure)
        vertical = cv.dilate(vertical, verticalStructure)
        # Show extracted vertical lines
        show_wait_destroy("vertical", vertical)

        # Inverse vertical image

        merge = cv.add(vertical ,horizontal)
        show_wait_destroy("merge", merge)

        rows_to_be_black = int(vertical.shape[0] // 4.5)  # length from top upto smaller grid

        table = merge.copy()
        head = merge.copy()

        table[:rows_to_be_black, :] = 0
        table[:, :5] = 0
        head[rows_to_be_black:, :] = 0
        head[:, :5] = 0

        kernel = np.ones((5,5), np.uint8) 
        table = cv.dilate(table, kernel)
        head = cv.dilate(head, kernel)

        # show_wait_destroy('org_gray---',org_gray)
        # show_wait_destroy('table---', table)
        # show_wait_destroy('head---', head)
        no_grid = cv.add(org_gray,table)
        no_grid = cv.add(no_grid,head)




        table,*xywh = self.closed_grid(table)
        show_wait_destroy('table---',table)
        show_wait_destroy('no_grid---',no_grid)
        table = self.complete_grid(table,no_grid,*xywh)
        inverted_table = self.inverted_table(table,*xywh)

        head,*xywh = self.closed_grid(head)
        inverted_head = self.inverted_table(head,*xywh)

        


        # show_wait_destroy("table", table)

        return self.img, org_gray,table,inverted_table,inverted_head,no_grid

    def inverted_table(self,table,*xywh):
        x,y,w,h = (xywh)
        # table[y:y+h,x:x+w] = 255 # only table area
        height,width = table.shape[:2]
        mask = np.zeros((height,width),dtype=np.uint8)
        mask[y:y+h,x:x+w] = 255
        show_wait_destroy("mask--", mask)
        table = cv.bitwise_not(table, mask=mask)
        show_wait_destroy("inverted_table", table)

        return table


    def distance(self,p1=None,p2=None):

        import math
        p1 = [4, 0]
        p2 = [6, 6]
        distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )

        print(distance)
        return distance


    def table_cell_list(self,table_bbs,img):

        image_list = []
        for row_tup in table_bbs:
            image_list.append(tuple([self.gray[y:y+h,x:x+w] for (x,y,w,h) in row_tup]))
            # image_list.append(tuple([img[y:y+h,x:x+w] for (x,y,w,h) in row_tup]))

        return tuple(image_list)
