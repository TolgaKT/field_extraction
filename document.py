import pytesseract
import pdf2image
from bs4 import BeautifulSoup as bs
import cv2 as cv
import os
import difflib
import numpy as np
from scipy.ndimage import interpolation as inter

class Document:

    hocr = ''
    parseResults = dict()
    imageList = []
    fields = []
    coordinates = ''

    def __init__(self,filePath):
        self.filePath = filePath

    def isPdf(self):
        fileName = self.filePath.split('/')[-1]
        #check if the provided file is a pdf
        if(fileName.split('.')[-1] == 'pdf'):
        #store pages of pdf as a list
            imageList = pdf2image.convert_from_path(self.filePath)
        #loop thorugh list
        for i in range(len(imageList)):
            #save the created images as jpg
            imageList[i].save(fileName.split('.')[0]+ '_' + str(i) + '.jpg','JPEG')
            self.imageList.append(fileName.split('.')[0]+ '_' + str(i) + '.jpg')

    def rotate(self,file):
        #find the best angle with the best score
        #based on the profile projection method
        def findScore(arr,angle):
            data = inter.rotate(arr, angle, reshape=False, order=0)
            hist = np.sum(data, axis=1)
            score = np.sum((hist[1:] - hist[:-1]) ** 2)
            return hist, score
        #read the file
        image = cv.imread(file)
        
        #converting the image to grayscale

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


        delta = 1
        limit = 5
        angles = np.arange(-limit, limit+delta, delta)
        scores = []
        for angle in angles:
            hist, score = findScore(gray, angle)
            scores.append(score)

        best_score = max(scores)
        best_angle = angles[scores.index(best_score)]
        data = inter.rotate(gray, best_angle, reshape=False, order=0)
        cv.imwrite(file,data)

    def ocr(self,file):
        fileName = file.split('/')[-1]
        hocr = pytesseract.image_to_pdf_or_hocr(fileName,extension='hocr')
        self.hocr = hocr

    def parseHocr(self,keyWords):
        #check if hocr is variable is empty
        if (self.hocr == ''):
            raise ValueError('OCR command must be executed before parsing')
        #check if keyWords parameter is of tuple
        if not isinstance(keyWords,tuple):
            raise ValueError('The keywords parameter must be a tuple')

        soup = bs(self.hocr,'html.parser')
        words = soup.find_all('span',class_='ocrx_word')
        # Loop through all the words and look for our search terms.
        # case sensitive!

        #key words that contains more than one keyword should be considered seperately
        multipleWords = [x for x in keyWords if len(x.split(' '))>1]
        keys = [x for x in keyWords if len(x.split(' '))==1]
        for i in range(len(words)):
            if(words[i].get_text() in keys or words[i].get_text().split(':')[0] in keys):
                #remove unnecessary elements
                #make it only include coordinates
                bbox = words[i]['title'].split(';')[0]
                bbox = bbox.split(' ')
                bbox = tuple([int(x) for x in bbox[1:]])
                #check the next bbox
                #this allows the program to both work for fields that are placed below the keyword or
                #field that are placed at the right of the key word
                #works by checking the y coordinates
                #if difference between y coordinates above a certain point we look below
                nextBbox = words[i+1]['title'].split(';')[0]
                nextBbox = nextBbox.split(' ')
                nextBbox = tuple([int(x) for x in nextBbox[1:]])
                self.parseResults[bbox] = abs(bbox[1] - nextBbox[1]) <15
        
        if(len(multipleWords)>0):
            for i in range(len(words)):
                for multiple in multipleWords:
                    #check both the first and second word in the keyword
                    #to avoid unwanted extractions
                    #using sequence matcher to make sure we dont miss the wanted keyword
                    if(difflib.SequenceMatcher(None, multiple.split(' ')[0],words[i].get_text()).ratio()>.5 and 
                    difflib.SequenceMatcher(None, multiple.split(' ')[1],words[i+1].get_text()).ratio()>.5):
                        bbox = words[i + len(multiple.split(' '))]['title'].split(';')[0]
                        bbox = bbox.split(' ')
                        bbox = tuple([int(x) for x in bbox[1:]])
                        nextBbox = words[i + len(multiple.split(' ')) + 2]['title'].split(';')[0]
                        #check the next bbox
                        #this allows the program to both work for fields that are placed below the keyword or
                        #field that are placed at the right of the key word
                        #works by checking the y coordinates
                        #if difference between y coordinates above a certain point we look below
                        nextBbox = nextBbox.split(' ')
                        nextBbox = tuple([int(x) for x in nextBbox[1:]])
                        self.parseResults[bbox] = abs(bbox[1] - nextBbox[1])<20
                            

    
    def returnFields(self,image):
        fields = []
        for coordinate,orientation in self.parseResults.items():
            img = cv.imread(image)
            cropped = img
            if(orientation):
                #crops the image properly regarding the position of the sought fields
                cropped = img[coordinate[1]-8:coordinate[3]+8,coordinate[2]:img.shape[1]-50]
            else:
                cropped = img[coordinate[1]-10:coordinate[3]+10,coordinate[0]-10:img.shape[1]-50]
            cv.imshow('cropped',cropped)
            cv.waitKey()
            gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
            thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

            # Blur and perform text extraction
            thresh = cv.GaussianBlur(thresh, (3,3), 0)
            data = pytesseract.image_to_string(thresh,config='--psm 6',lang='tur')
            #remove unwanted characters
            data = data.split('* ')[-1]
            data = os.linesep.join([s for s in data.splitlines() if s])
            fields.append(data)
        
        self.fields = fields
        #return self.fields
        