"""
Yusuf Can Anar

1 - Kırmızıların en parlak olduğu yeri tespit etme 
2 - Bu tespit edilen yerlerden çubukları bulma 
3 - bulunan çubuklardan sadece ilgilendiğimizi tespit etme
4 - bir önceki adımda bulunan çubuğun köşe koordinatlarını tespit etme ve
    bu köşelerin arasındaki öklit uzaklığını hesaplama


"""
import numpy as np
import cv2
import copy

def viewImage(image): # shows the inputted images
    print("To see the next image press any key\n")
    cv2.namedWindow('Steps',cv2.WINDOW_NORMAL)
    cv2.imshow('Steps',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
       
image = cv2.imread('akkor.jpeg') #Original Image
print("Original Image")
viewImage(image)
corner_obj = copy.deepcopy(image)

img = cv2.bilateralFilter(image,9,75,75) # denoise doesn't necessary
print("Denoise")
viewImage(img)


# Step 1
hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) #HSV Image
print("HSV")
viewImage(hsv_img)

#To be able to create a mask we convert the image to HSV
#So we can create a mask that has red color instensity
red_low = np.array([0,100,100])
red_high = np.array([30,255,255])
curr_mask = cv2.inRange(hsv_img, red_low, red_high)
#If curent mask got an instensity greater than 0
# Then hsv_img 's same element becomes [0,255,255] this color directly
hsv_img[curr_mask > 0] = ([0,255,255]) #Mask

print("Red Mask of HSV")
viewImage(curr_mask)

print("Masked HSV")
viewImage(hsv_img)

# Step 2
RGB_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
gray = cv2.cvtColor(RGB_img, cv2.COLOR_RGB2GRAY) # Gray Image
print("Gray Image")
viewImage(gray)

ret,threshold = cv2.threshold(gray,234,255,0) #Threshold Image
print("Threshold")
viewImage(threshold)

def blue_mask(image):
    blue = np.array([120,255,255])
    mask_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b_mask = cv2.inRange(mask_image,blue,blue)#Blue mask
    viewImage(b_mask)
    return b_mask   

contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
def All_contours(contours):
    """This function displays all the contours in the blurred image"""
    _img = copy.deepcopy(img)
    cv2.drawContours(_img,contours,-1,(255,0,0),3) # Draws Contours
    print("All contours are found")
    viewImage(_img)
    
new_contours = []
def remove_small_regions(contours):
    """This Function calculates all contour areas, finds their average area to eliminate the unwanted small regions"""
    global new_contours
    contours_index = len(contours)
    
    contours_area = np.array([cv2.contourArea(contours[i]) for i in range(contours_index)]) # area array
    mean_area = np.sum(contours_area) / contours_index # average area

    cnt = np.array([ contours_area[i] > mean_area for i in range(contours_index)]) 
    for i in range(contours_index):
        if cnt[i] == True:
            new_contours.append(contours[i])# list of new contours that have greater area than the average area
            
    
    cv2.drawContours(img,new_contours,-1,(255,0,0),3)
    print("Only big contours are left")
    viewImage(img) # Image with only big contours
    
    print("Blue Mask to have a clear sight of big contours")
    blue_mask(img) # binary image without small regions
      
def find_first_bar():
    """This function detects only the first bar on the image"""
    global new_contours
    # y düzleminde(row) en düsük koordinatlara sahip contouru gösterir
    # Step 3
    y = []
    min_y = []
    latest_min = img.shape[0]
    
    for i in range(len(new_contours)):
        
        a = new_contours[i] # contours adlı listedeki 3D arrayleri a ya atar 
        boyut = a.shape
        
        for j in range(boyut[0]):
            row = a[j][0][1] # 3D array deki y elementini row değişkenine atar
            y.append(row)    # y listesinde y koordinatlarını listeler
        
        min_row = np.amin(y) # y listesindeki en düşük değeri min_row'a atar
        min_y.append(min_row)# min_y listesine a[i] arrayinin en düşük y değerini ekler
        min_of_all = np.amin(min_y) # min_of_all değişkenine her döngüde elde edilen min_y listesinin içindeki en küçük değer atanır
        
        if min_of_all < latest_min: # en küçük değeri sahip olduğumuzla karşılaştırır eğer daha küçükse yeni en küçük değeri atar
                                    # ve koordinatları y-axisinde en küçük olan contourun indexini tespit eder
            latest_min = min_of_all # böylece görüntünün akışında her zaman ilk akkor barı tespit edebiliriz
            index = i
        

    cv2.drawContours(image,new_contours, index,(255,0,0),3)
    cv2.fillConvexPoly(image,new_contours[index],[255,0,0])
    print("Detection of wanted contour")
    viewImage(image)#Final Image
    

def finding_corners_and_euclidean_distance():
    
    print("Mask for corner detection")
    b_mask = blue_mask(image)
    
    #Specifying maximum number of corners as 1000
    # 0.01 is the minimum quality level below which the corners are rejected
    # 7 is the minimum euclidean distance between two corners
    corners_img = cv2.goodFeaturesToTrack(b_mask,1000,0.01,7)
    
    for corners in corners_img:
       
        x,y = np.ravel(corners)
        
        #Circling the corners in blue
        cv2.circle(corner_obj,(x,y),1,[255,0,0],-1)
    print("corner points of the object")
    viewImage(corner_obj)
    print("clear sight of the points with the blue mask")
    blue_mask(corner_obj)
    
    # get the min area rect
    rect = cv2.minAreaRect(corners_img) 
    box = cv2.boxPoints(rect)
    # convert all coordinates floating point values to int
    box = np.int0(box)
    # draw a rectangle to fit the object
    cv2.drawContours(corner_obj, [box], 0, (255, 0, 120),2)
    print ("fitting rectengular illustration of the bar")
    viewImage(corner_obj)# fitting rectengular illustration of the bar
    
    x,y,w,h = cv2.boundingRect(corners_img)
    euclidean_distance = np.sqrt((w**2) + (h**2))
    print ("euclidean distance of the bar between coordinates of",(x,y),
           "and",(x+w,y+h),"is", euclidean_distance)
        
    
All_contours(contours)
remove_small_regions(contours)
find_first_bar()
finding_corners_and_euclidean_distance()








    
