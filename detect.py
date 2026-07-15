import cv2 

class IDataExtractor:
    def __init__(self):
        pass
    
    def detect_card(self,img):
        gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # No need to colors 
        blur=cv2.GaussianBlur(gray,(5,5),0)
        edges=cv2.Canny(blur,30,150)
        contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img,contours,-1,(255,0,0),1)
        
        max_contour =max(contours,key=cv2.contourArea)
        x, y, w, h=cv2.boundingRect(max_contour)
        id_card = img[y:y+h,x:x+w]
        return id_card
    
    def define_regions ():
        pass
    
    
    def process_image(self,img):
        id_card = self.detect_card(img)
        regions = self.define_regions(id_card)
    
id_extractor =IDataExtractor()
img = cv2.imread('image.jpg')
output = id_extractor.process_image(img) 


cv2.imshow('frame',output)
cv2.waitKey(0)
cv2.destroyAllWindows()


