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

        self.img = img
        self.gray = '' 

    def fix_page_orientation(self):
            
            osd = pytesseract.image_to_osd(self.img)
            information =  osd.splitlines()
            rotations = information[2]
            _,angle = rotations.split(':')
            angle = int(angle)
            print(f"angle -- {angle}")

            ###############################################
            # grab the dimensions of the image and then determine the
            # center
            (h, w) = self.img.shape[:2]
            (cX, cY) = (w // 2, h // 2)
        
            # grab the rotation matrix (applying the negative of the
            # angle to rotate clockwise), then grab the sine and cosine
            # (i.e., the rotation components of the matrix)
            M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
        
            # compute the new bounding dimensions of the image
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
        
            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY
        
            # perform the actual rotation and return the image
            self.img = cv.warpAffine(self.img, M, (nW, nH))

            self.set_image(self.img)
            
            # return self.img

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
        # cv.imshow('gray',self.gray)

        org_gray = gray.copy()
        # Show gray image
        show_wait_destroy("gray", gray)
        
        # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
        gray = cv.bitwise_not(gray)
        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                    cv.THRESH_BINARY, 15, -2)
        # Show binary image

        # testing
        kernel = np.ones((5,5), np.uint8) 
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

        no_grid = cv.add(org_gray,table)
        no_grid = cv.add(no_grid,head)

        # # Step 1
        # edges = cv.adaptiveThreshold(table, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, -2)
        # show_wait_destroy("edges", edges)
        # # Step 2
        # kernel = np.ones((2, 2), np.uint8)
        # edges = cv.dilate(edges, kernel)
        # show_wait_destroy("dilate", edges)
        # table = edges

        # cv2.erode(img, kernel, iterations=1)
        # table = cv.erode(table, kernel)


        table,*xywh = self.closed_grid(table)
        table = self.complete_grid(table,no_grid,*xywh)

        inverted_table = self.inverted_table(table,*xywh)


        # show_wait_destroy("table", table)

        return org_gray,table,inverted_table,head,no_grid


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

        return tuple(image_list)
