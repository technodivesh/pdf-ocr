import cv2 as cv
import numpy as np
from read_pdf import ReadPdf
import settings
import glob


DEBUG = True

def show_wait_destroy(winname, img):
    if not DEBUG: return
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

class Image2text:

    def __init__(self,img):

        self.img = img


    def get_amt(self):

        print ("get_amt")
        

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
        self.gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)


    def findcnt(self):

        h,w = self.img.shape[:2]
        # print(w,h)
        inv_img = cv.bitwise_not(self.gray)
        half_img = inv_img[ int(h/2):,: ]
        half_img_col = self.img[ int(h/2):,: ]
        thresh = cv.threshold(half_img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        show_wait_destroy('thresh',thresh)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        cv.drawContours(half_img,contours)

        print(len(contours))






if __name__ == "__main__":


    INPUT_DIR = settings.INBOX
    # pdfs = glob.glob(f'{INPUT_DIR}/*229.pdf')
    pdfs = glob.glob(f'{INPUT_DIR}/*22123708.pdf')
    

    for pdf in pdfs[:1]:
        print("pdf--",pdf)
        pdf_obj = ReadPdf(pdf)

        for pg_num in range(pdf_obj.num_of_pages())[:1]:

            img,status = pdf_obj.read_page(pg_num)
            show_wait_destroy('img', img)

            Image2textObj = Image2text(img)
            Image2textObj.fix_page_orientation()
            fixed_img = Image2textObj.get_image()
            show_wait_destroy('fixed_img', fixed_img)

            Image2textObj.findcnt()




