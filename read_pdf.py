#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__      = "Divesh Chandolia"

from PyPDF2 import PdfFileReader
from pdf2image import convert_from_path
import settings
import os
import numpy as np


class ReadPdf():

    def __init__(self,pdf):

        self.pdf = pdf
        self.pdfNameOnly = os.path.splitext(os.path.basename(pdf))[0]

    def num_of_pages(self):

        return PdfFileReader(open(self.pdf,'rb')).getNumPages()

    def read_page(self,page_number):
        "It reads one page at a time and retun image as numpy array"

        print("pdf to image ", page_number)
        
        # outbox = os.path.join(settings.OUTBOX,self.pdfNameOnly)
        # if not os.path.exists(outbox):
        #     print(f"creating DIR - {outbox}")
        #     os.makedirs(outbox)

        page_list = convert_from_path(self.pdf,
                                dpi=settings.DPI,
                                fmt="png",
                                first_page=page_number,
                                last_page=page_number,
                                # grayscale=True,
                                size=None,
                                transparent=True,
                                strict=True,
                                # output_folder = outbox,
                                # output_file=f"{self.pdfNameOnly}-{page_number}.png",
                                )

        print("pdf to image - end")
        page = np.array(page_list[0].convert('RGB')) # BGR image in opencv
        return page,True



    def read_pages(self):
        "It reads all pages from pdf and return images as numpy aaray"
        print("reading all pages")
        
        # outbox = os.path.join(settings.OUTBOX,self.pdfNameOnly)
        # if not os.path.exists(outbox):
        #     print(f"creating DIR - {outbox}")
        #     os.makedirs(outbox)

        print(f"max thread-{8 if self.num_of_pages > 8 else self.num_of_pages}")

        pages_list = convert_from_path(self.pdf,
                                dpi=settings.DPI,
                                fmt="png",
                                first_page=1,
                                last_page=8,
                                # grayscale=True,
                                size=None,
                                transparent=True,
                                strict=True,
                                # output_folder = outbox,
                                # output_file=f"{self.pdfNameOnly}-{page_number}.png",
                                thread_count=8 if self.num_of_pages > 8 else self.num_of_pages, # max 8 thread else equal to no. of pages
                                )

        pages_list = [np.array(page.convert('RGB')) for page in pages_list ] # converting from PIL to cv image(numpy array) 
        return pages_list


if __name__ == '__main__':

    import time

    path = '/home/root1/AllErrorFiles/testfiles/22123708.pdf'
    # path = "/home/root1/AllErrorFiles/testfiles/330_BCTN2018111511001229 (002).pdf"
    pdf_obj = ReadPdf(path)
    n = pdf_obj.num_of_pages()
    print(f"total pages-{n}")
    
    for pg_num in range(1,n):
        start_time = time.time()
        print(pg_num)
        pages = pdf_obj.read_page(pg_num)
        print(time.time() - start_time)

    # start_time = time.time()
    # pages = pdf_obj.read_pages()
    # # print(pages)
    # print(time.time() - start_time)









