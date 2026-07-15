import cv2


# def limeted(img):
#         gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # No need to colors
#         for i ,j in gray:
#             if gray[i,j] > 200 :
#                gray[i,j] = 200
               
#         return gray       
               
     
def limeted(img):
        gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # No need to colors
        for i in range(0,gray.shape[0]):
            for j in range(0,gray.shape[1]):
                if gray[i,j] > 200 :
                    gray[i,j] = 185
               
        return gray       
                    
img = cv2.imread('image.jpg')
# # output = id_extractor.process_image(img) 
img = limeted(img)
cv2.imshow("frame",img)  
          
cv2.waitKey(0)
cv2.destroyAllWindows()