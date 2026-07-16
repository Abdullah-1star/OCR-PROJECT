import cv2 
from deepface import DeepFace  

class IDataExtractor:
    def __init__(self):
        pass
    
    def detect_card(self,img):
        gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # No need to colors 
        blur=cv2.GaussianBlur(gray,(5,5),0)
        edges=cv2.Canny(blur,30,150)
        contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img,contours,-1,(255,0,0),1)
        
        max_contour =max(contours,key=cv2.contourArea)
        x, y, w, h=cv2.boundingRect(max_contour)
        id_card = img[y:y+h,x:x+w]
        return id_card
    
    
    
    
    
    
    
    # def define_regions(self, id_card):
    #     gray = cv2.cvtColor(id_card, cv2.COLOR_BGR2GRAY)
    #     blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #     _, segmented = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #     contours, _ = cv2.findContours(segmented, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #     # cv2.drawContours(id_card, contours, -1, (0, 0, 255), 1)
    #     photo, coords = self.get_photo(id_card, contours)
    #     angle, id_card = self.get_right_position(id_card, coords)

    #     while angle!=None:
    #         gray = cv2.cvtColor(id_card, cv2.COLOR_BGR2GRAY)
    #         blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #         _, segmented = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #         contours, _ = cv2.findContours(segmented, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #         # cv2.drawContours(id_card, contours, -1, (0, 0, 255), 1)
    #         photo, coords = self.get_photo(id_card, contours)
    #         angle, id_card = self.get_right_position(id_card, coords)

    #     return id_card, photo
    
    
    # def define_regions (self,id_card):
    #     gray= cv2.cvtColor(id_card, cv2.COLOR_BGR2GRAY) # No need to colors 
        
    #     clahe = cv2.createCLAHE(
    #              clipLimit=0.3,
    #             tileGridSize=(8,8)
    #             )

    #     gray = clahe.apply(gray)
        
    #     blur=cv2.GaussianBlur(gray,(3,3),0)
    #     _,segmented = cv2.threshold(blur,100,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     contours,_=cv2.findContours(segmented,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    #     # cv2.drawContours(id_card,contours,-1,(0,0,255),1)
    #     face =self.get_face(id_card,contours)
    #     return face
    
    def define_regions(self, id_card):

        while True:

            gray = cv2.cvtColor(id_card, cv2.COLOR_BGR2GRAY)

            clahe = cv2.createCLAHE(
                clipLimit=0.3,
                tileGridSize=(8, 8)
            )

            gray = clahe.apply(gray)

            blur = cv2.GaussianBlur(gray, (3, 3), 0)

            _, segmented = cv2.threshold(
                blur,
                100,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            contours, _ = cv2.findContours(
                segmented,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # cv2.drawContours(id_card, contours, -1, (0,0,255), 1)

            face, coords = self.get_face(id_card, contours)

            angle, id_card = self.get_right_position(id_card, coords)

            if angle is None:
                break

        return id_card, face
    
    
    def get_face(self,id_card,contours):
        contours = list(contours)
        max_idx=max(range(len(contours)),key= lambda i: cv2.contourArea(contours[i])) 
        contours.pop(max_idx)
        max_idx=max(range(len(contours)),key= lambda i: cv2.contourArea(contours[i])) 
        max_contour=contours[max_idx]
        x, y, w, h=cv2.boundingRect(max_contour) 
        y_strat =int(y - 1.254 * h)
        y_end =y+h 
        x_start = x
        x_end=int(x+0.9*w)
        face=id_card[y_strat : y_end , x_start :x_end]
       
        return face , (x_start , y_strat , x_end , y_end)
    
    def verify_live(self,photo):
        cap=cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print( 'cant open the camere ' )
            return False
        
        while True:
            
            ret,frame= cap.read()
            
            if not ret:
                print ( 'no frame detected' )
                cap.release()
                cv2.destroyAllWindows()
                return False
            
            cv2.imshow( 'frame' , frame )
            
            key = cv2.waitKey(1) & 0xFF
            print(key)
            if key == ord('c'):
                break
            
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return False
         
        verified = DeepFace.verify(
                        frame,
                        photo,
                        detector_backend="retinaface"
                        )  
        cap.release()
        cv2.destroyAllWindows()
        return verified['verified']
    
    def get_right_position(self, id_card, coords):
        xmin, ymin, xmax, ymax = coords
        angle = None
        shape = id_card.shape
        if xmin > abs(xmax- shape[1]) and ymin < abs(ymax-shape[0]):
            print("Rotated Right")
            angle = 90
            id_card = cv2.rotate(id_card, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif xmin > abs(xmax- shape[1]) and ymin > abs(ymax-shape[0]) :
            print("Flipped")
            angle = 180
            id_card = cv2.rotate(id_card, cv2.ROTATE_180)    

        elif ymin > abs(ymax-shape[0]):
            print("Rotated Left")
            angle = -90
            id_card = cv2.rotate(id_card, cv2.ROTATE_90_CLOCKWISE)
        
        return angle, id_card
    
    def process_image(self,img):
        id_card = self.detect_card(img)
        id_card, photo = self.define_regions(id_card) 
        verified =self.verify_live(photo)
        print(verified)
        return photo
    
id_extractor =IDataExtractor()
img = cv2.imread('image2.jpeg')
output = id_extractor.process_image(img) 


cv2.imshow('frame',output)
cv2.waitKey(0)
cv2.destroyAllWindows()



