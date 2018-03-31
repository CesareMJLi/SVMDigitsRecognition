'''

Created Mar 31 2018

@ Author Mingju Li

'''

#----------------------Imports---------------------------

import numpy as np
from scipy.misc.pilutil import imresize
import cv2 #version 3.2.0
from skimage.feature import hog
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

#----------------------Constants delaration---------------------------

TRAIN_IMG = 'train_image.png' 
INPUT_IMG = 'input_image.png'

IMG_HEIGHT = 28
IMG_WIDTH = 28

# The given train image, each number should have a size of 50 * 10 pixels.
# each row have 100 samples and each column have 50 elements
# so totally there are 5000 samples
# 0 is from 0 to 499
# 1 is from 500 to 999
# ...

#----------------------Functions of training---------------------------

def pixels_to_hog_20(img_array):
    hog_featuresData = []
    for img in img_array:
        fd = hog(img, 
                 orientations=10, 
                 pixels_per_cell=(5,5),
                 cells_per_block=(1,1), 
                 visualise=False)
        hog_featuresData.append(fd)
    hog_features = np.array(hog_featuresData, 'float64')
    return np.float32(hog_features)

#this function processes a custom training image
#see example : custom_train.digits.jpg
#if you want to use your own, it should be in a similar format
def load_digits(img_file):
    train_data = []
    train_target = []
    start_class = 0
    im = cv2.imread(img_file)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    imgray = (255-imgray)
    # this above is added to make sure the imgray is turned into a white at bottom and black fonts style
    plt.imshow(imgray)
    kernel = np.ones((5,5),np.uint8)
    
    ret,thresh = cv2.threshold(imgray,127,255,0)   
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    thresh = cv2.dilate(thresh,kernel,iterations = 1)
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    
    _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    digits_rectangles = get_digits(contours,hierarchy)  
    #rectangles of bounding the digits in user image
    
    #sort rectangles accoring to x,y pos so that we can label them
    digits_rectangles.sort(key=lambda x:get_contour_precedence(x, im.shape[1]))
    
    for index,rect in enumerate(digits_rectangles):
        # >>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        # >>> list(enumerate(seasons))
        # [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
        x,y,w,h = rect
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        im_digit = imgray[y:y+h,x:x+w]
        im_digit = (255-im_digit)
        im_digit = imresize(im_digit,(IMG_WIDTH, IMG_HEIGHT))

        train_data.append(im_digit)
        train_target.append(start_class%10)

        if index>0 and (index+1) % 500 == 0:
            # since each line there is only 10 numbers for 0/1/2/...
            start_class += 1
    cv2.imwrite("training_box_overlay.png",im)
    return np.array(train_data), np.array(train_target)

def get_contour_precedence(contour, cols):
    return contour[1] * cols + contour[0]  #row-wise ordering

# given contours and hierarchy, return a list of final bounding rectanges with respond to the numbers
def get_digits(contours, hierarchy):
    hierarchy = hierarchy[0]
    # now the hierarchy is the root hierarchy in the diagram
    bounding_rectangles = [cv2.boundingRect(ctr) for ctr in contours]   
    # x, y, w, h = cv2.boundingRect(cnt)

    final_bounding_rectangles = []
    #find the most common heirarchy level - that is where our digits's bounding boxes are
    u, indices = np.unique(hierarchy[:,-1], return_inverse=True)
    # [:-1] is from the first to the last

    most_common_heirarchy = u[np.argmax(np.bincount(indices))]

    print("The most common heirarchy is ")
    print(most_common_heirarchy)
    
    for r,hr in zip(bounding_rectangles, hierarchy):
        # zip is a wrapping func
        # here r is the rectangle and hr is its relative hierarchy
        x,y,w,h = r

        #this could vary depending on the image you are trying to predict
        #we are trying to extract ONLY the rectangles with images in it (this is a very simple way to do it)
        #we use heirarchy to extract only the boxes that are in the same global level - to avoid digits inside other digits
        #ex: there could be a bounding box inside every 6,9,8 because of the loops in the number's appearence - we don't want that.
        #read more about it here: https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html

        if  (w <= 50) and (h <= 100) and hr[3] == most_common_heirarchy: 
            # hr[3] is the parent contours
            # here I think the most common heirarchy should be -1
            final_bounding_rectangles.append(r)    

    return final_bounding_rectangles

#----------------------Functions of testing---------------------------

def proc_user_img(img_file, model):
    print('loading "%s for digit recognition" ...' % img_file)
    im = cv2.imread(img_file)    

    blank_image = np.zeros((im.shape[0],im.shape[1],3), np.int32)
    # blank_image = np.zeros((img.shape[0],img.shape[1],3),dtype=int)
    # the above line is my original solution
    # while it cause a problem in the following output there is an uncompatible error
    # switch to another data type -- np.int32 could solve this problem
    blank_image.fill(255)

    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    plt.imshow(imgray)
    kernel = np.ones((5,5),np.uint8)
    
    ret,thresh = cv2.threshold(imgray,127,255,0)   
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    thresh = cv2.dilate(thresh,kernel,iterations = 1)
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    # the above operation is to do the open/close on the image, to obtain the main shape in the image 
    
    _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # https://blog.csdn.net/sunny2038/article/details/12889059
    # https://blog.csdn.net/jfuck/article/details/9620889


    digits_rectangles = get_digits(contours,hierarchy)  
    #rectangles of bounding the digits in user image
    
    for rect in digits_rectangles:
        x,y,w,h = rect
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        # cv2.rectangle(img,(380,0),(511,111),(255,0,0),3)
        # the parameter to be decided are the two points of the rectangle, color and type of lines
        im_digit = imgray[y:y+h,x:x+w]
        im_digit = (255-im_digit)
        im_digit = imresize(im_digit,(IMG_WIDTH ,IMG_HEIGHT))

        hog_img_data = pixels_to_hog_20([im_digit])  
        pred = model.predict(hog_img_data)
        cv2.putText(im, str(int(pred[0])), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        cv2.putText(blank_image, str(int(pred[0])), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

    plt.imshow(im)
    cv2.imwrite("original_overlay.png",im) 
    cv2.imwrite("final_digits.png",blank_image) 
    cv2.destroyAllWindows() 

#----------------------Class---------------------------

# The main class used deploy supportive vector machine.

class SVM_MODEL():
    def __init__(self, num_feats, C = 1, gamma = 0.1):
        self.model = cv2.ml.SVM_create()
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setKernel(cv2.ml.SVM_RBF) #SVM_LINEAR, SVM_RBF
        self.model.setC(C)
        self.model.setGamma(gamma)
        self.features = num_feats

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        results = self.model.predict(samples.reshape(-1,self.features))
        return results[1].ravel()

#----------------------Training---------------------------

digits, labels = load_digits(TRAIN_IMG)

print('train data shape',digits.shape)
print('test data shape',labels.shape)

digits, labels = shuffle(digits, labels, random_state=256)
train_digits_data = pixels_to_hog_20(digits)
train_samples, test_samples, train_labels, test_labels = train_test_split(train_digits_data, labels, test_size=0.33, random_state=42)

#----------------------Testing---------------------------

# print(train_digits_data.shape)

model = SVM_MODEL(num_feats = train_digits_data.shape[1])
model.train(train_samples, train_labels)
preds = model.predict(test_samples)
print('Accuracy: ',accuracy_score(test_labels, preds))

model = SVM_MODEL(num_feats = train_digits_data.shape[1])
model.train(train_digits_data, labels)
proc_user_img(INPUT_IMG, model)

#------------------------------------------------------------------------------