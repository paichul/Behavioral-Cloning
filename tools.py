import cv2

def preprocess(image):
  yuv = cv2.cvtColor(image,cv2.COLOR_BGR2YUV) #convert from BGR to YUV
  hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
  yuv[:,:,2] = hsv[:,:,2] #replace yuv's v channel with hsv's v channel
  image = image[50:-20, 30:-30,:] #cropping
  return image
