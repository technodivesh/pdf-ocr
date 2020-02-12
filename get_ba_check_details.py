import cv2 as cv
import numpy as np
from read_pdf import ReadPdf
import settings
import glob
# import imutils
from pytesseract import image_to_string



DEBUG = False

def show_wait_destroy(winname, img):
    if not DEBUG: return
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

class Cheque:

    def __init__(self,img):

        self.img = img
        self.gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    def fix_page_orientation(self):

        #############################################################
        # rotate the image to deskew it
        angle = -4
        (h, w) = self.img.shape[:2]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        self.set_image(cv.warpAffine(self.img, M, (w, h),
            flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE))
        show_wait_destroy("self.img--",self.img)
        ##############################################################

        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        gray = cv.bitwise_not(gray)
        
        # threshold the image, setting all foreground pixels to
        # 255 and all background pixels to 0
        thresh = cv.threshold(gray, 0, 255,
            cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

        
        # grab the (x, y) coordinates of all pixel values that
        # are greater than zero, then use these coordinates to
        # compute a rotated bounding box that contains all
        # coordinates
        coords = np.column_stack(np.where(thresh > 0))
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
        (h, w) = self.img.shape[:2]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        self.set_image(cv.warpAffine(self.img, M, (w, h),
            flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE))


    def get_image(self):
        return self.img

    def set_image(self,img):
        self.img = img
        self.gray = cv.cvtColor(self.img,cv.COLOR_BGR2GRAY)

    def get_check_amt(self,img,img_col):

        img_col[img_col[:,:,:]>200] = 255

        # show_wait_destroy('check amt2--',img)
        blurred = cv.GaussianBlur(img, (5, 5), 0)
        # show_wait_destroy('blurred',blurred)

        thresh = cv.threshold(blurred, 60, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        # show_wait_destroy('thresh',thresh)

        contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv.contourArea, reverse=True)[1:3]
        # print(len(contours))
        dollerCentCoord = (cv.boundingRect(c) for c in contours)
        dollerCentImgs = (img_col[y+2:y+h-2,x+2:x+w-2] for x,y,w,h in dollerCentCoord)
        dollerCentStr = (image_to_string(cnt_img) for cnt_img in dollerCentImgs)
        dollerCentInt = ("".join(list(filter(lambda x: x.isdigit(),value))) for value in dollerCentStr)
        self.check_amt = ".".join(list(dollerCentInt))

    def find_cheque_details(self):

        self.img_bin =  cv.bitwise_not(self.gray)

        # scale_percent = 50 # percent of original size
        # width = int(self.img_bin.shape[1] * scale_percent / 100)
        # height = int(self.img_bin.shape[0] * scale_percent / 100)
        # dim = (width, height)
        # # resize image
        # resized = cv.resize(self.img_bin, dim, interpolation = cv.INTER_AREA)
        # resized_col = cv.resize(self.img, dim, interpolation = cv.INTER_AREA)

        resized = self.img_bin
        resized_col = self.img

        show_wait_destroy('resized',resized)

        blurred = cv.GaussianBlur(resized, (5, 5), 0)
        show_wait_destroy('blurred',blurred)

        thresh = cv.threshold(blurred, 60, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        # thresh = cv.GaussianBlur(resized, (5, 5), 0)
        show_wait_destroy('thresh',thresh)

        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # print(len(contours))
        # find the biggest countour (c) by the area
        c = max(contours, key = cv.contourArea)
        x,y,w,h = cv.boundingRect(c) # check amount cell

        # cv.drawContours(resized_col, [c], 0, (0,255,0), 1)

        # for check amt
        self.check_amt_img = resized_col[y:y+h,x:x+w]
        # text = image_to_string(self.check_amt_img)
        # print(text)
        show_wait_destroy('check amt--',self.check_amt_img)
        self.get_check_amt(resized[y:y+h,x:x+w], resized_col[y:y+h,x:x+w])

        # for check number
        y = y - h
        x = x + int(w * 10 /100)
        check_number_img = resized_col[y:y+h,x:x+w]
        self.check_number = image_to_string(check_number_img)
        # print(text)
        # show_wait_destroy('check No--',check_number_img)

        # for check date
        # y = y - h
        x = x - w
        check_date_img = resized_col[y:y+h,x:x+ w - (int(w * 25 /100)) ]
        self.check_date = image_to_string(check_date_img)
        # print(text)
        # show_wait_destroy('Date --',check_date_img)

        # for check address
        x = x - w*3
        w = w*3
        y = y + h*2
        h = h*2
        check_add_img = resized_col[y:y+h,x:x+w]
        self.check_add = image_to_string(check_add_img)
        # print(text)
        # show_wait_destroy('Date --',check_add_img)




        # coef_y = self.img.shape[0] / resized.shape[0]
        # coef_x = self.img.shape[1] / resized.shape[1]

        # c[:, :, 0] = c[:, :, 0] * coef_x
        # c[:, :, 1] = c[:, :,  1] * coef_y
        # cv.drawContours(self.img, c, -1, (0, 255, 0), 2)
        # show_wait_destroy('self.img',self.img)
        # cv.imwrite('selfimg.png',self.img)



        # coef_y = img_orig.shape[0] / img_resized.shape[0]
        # coef_x = img_orig.shape[1] / img_resized.shape[1]

        # for contour in contours:
        #     contour[:, :, 0] = contour[:, :, 0] * coef_x
        #     contour[:, :, 1] = contour[:, :,  1] * coef_y

        #     cv2.drawContours(img_orig, contour, -1, (0, 255, 0), 2)

    @property
    def amount(self):
        return self.check_amt

    @property
    def number(self):
        return self.check_number

    @property
    def date(self):
        return self.check_date

    @property
    def address(self):
        return self.check_add

    class ShapeDetector:
        def __init__(self):
            pass
    
        def detect(self, c):
            # initialize the shape name and approximate the contour
            shape = "unidentified"
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)

            # if the shape is a triangle, it will have 3 vertices
            if len(approx) == 3:
                shape = "triangle"
    
            # if the shape has 4 vertices, it is either a square or
            # a rectangle
            elif len(approx) == 4:
                # compute the bounding box of the contour and use the
                # bounding box to compute the aspect ratio
                (x, y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)
    
                # a square will have an aspect ratio that is approximately
                # equal to one, otherwise, the shape is a rectangle
                shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    
            # if the shape is a pentagon, it will have 5 vertices
            elif len(approx) == 5:
                shape = "pentagon"
    
            # otherwise, we assume the shape is a circle
            else:
                shape = "circle"
    
            # return the name of the shape
            return shape





if __name__ == "__main__":


    INPUT_DIR = settings.INBOX
    # pdfs = glob.glob(f'{INPUT_DIR}/*229.pdf')
    pdfs = glob.glob(f'{INPUT_DIR}/22128057.pdf')
    

    for pdf in pdfs[:1]:
        print("pdf--",pdf)
        pdf_obj = ReadPdf(pdf)

        for pg_num in range(pdf_obj.num_of_pages())[:1]:

            img,status = pdf_obj.read_page(pg_num)
            show_wait_destroy('img', img)

            cheque = Cheque(img)
            # cheque.fix_page_orientation()
            # fixed_img = cheque.get_image()
            # show_wait_destroy('fixed_img', fixed_img)

            cheque.find_cheque_details()

            print('--------------------')
            print(cheque.amount)
            print(cheque.number)
            print(cheque.date)
            print(cheque.address)




