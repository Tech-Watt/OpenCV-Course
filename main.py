import cv2 

# MODULE ONE

# ----------------------------------------------------------------
# Loading and displaying images
# ----------------------------------------------------------------
# image = cv2.imread(filename='car.jpg')
# cv2.imshow(winname='frame',mat=image)
# cv2.waitKey(0)


# ----------------------------------------------------------------
# Resizing images
# ----------------------------------------------------------------
# image = cv2.imread(filename='car.jpg')
# width,height,color_channel = image.shape
# print(f'width: {width}, height:{height} color_channel:{color_channel}')
# resized_image = cv2.resize(image,(500,500))
# cv2.imshow(winname='resized_image',mat=resized_image)
# cv2.imshow(winname='frame',mat=image)
# cv2.waitKey(0)


# ----------------------------------------------------------------
# How to convert images to grayscale
# ----------------------------------------------------------------
# image = cv2.imread(filename='car.jpg')

# width,height,color_channel = image.shape

# print(f'width: {width}, height:{height} color_channel:{color_channel}')
# resized_image = cv2.resize(image,(500,500))

# gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# cv2.imshow(winname='resized_image',mat=resized_image)

# cv2.imshow(winname='gray_image',mat=gray_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()



# MODULE TWO

# ----------------------------------------------------------------
# Drawing shapes on images
# ----------------------------------------------------------------
# image = cv2.imread(filename='soccer.jpg')

# # circle
# cv2.circle(image,(300,80),60,(255,0,0),-1)

# # rectangle
# cv2.rectangle(image,(60,60),(50,100),(0,0,255),4)

# # line
# cv2.line(image,(0,0),(150,150),(255,0,0),4)

# # text
# cv2.putText(image,'OPENCV',(350,90),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)

# # elipse
# cv2.ellipse(image,(255,255),(50,100),0,0,300,(255,0,0),5)

# # arrow line

# cv2.arrowedLine(image,(100,100),(300,200),(0,255,0),4)


# cv2.imshow(winname='frame',mat=image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# Module 3
# ----------------------------------------------------------------
# working with video files in OpenCV
# ----------------------------------------------------------------
# cap = cv2.VideoCapture('dance1.mp4')

# width = cap.get(3)
# height = cap.get(4)

# print(f'width: {width}, height:{height}')

# codec = cv2.VideoWriter.fourcc(*'mp4v')
# size = (640,480)

# writer = cv2.VideoWriter('output.mp4',codec,30,size)

# while True:
#     ret,frame = cap.read()
#     if not ret:
#         cap = cv2.VideoCapture('dance1.mp4')
#         continue

#     frame = cv2.resize(frame,(640,480))

#     writer.write(frame)

#     cv2.imshow(winname='frame',mat=frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# cap.release()
# cv2.destroyAllWindows()



# Module 4
# ----------------------------------------------------------------
# Image transformations
# ----------------------------------------------------------------
# image = cv2.imread('soccer.jpg')
# width, height,channel = image.shape

# cropped_image = image[50:300,50:300]

# angle = 45

# center = (width//2, height//2)
# rotate_matrix = cv2.getRotationMatrix2D(center,angle,1)
# rotated_image = cv2.warpAffine(image,rotate_matrix,(width, height))

# cv2.imshow('rotated_image', rotated_image)
# cv2.imshow('frame', image)
# cv2.imshow('cropped_image', cropped_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Module 5
# ----------------------------------------------------------------
# How to blur and detect edges in images
# ----------------------------------------------------------------

# image = cv2.imread(filename='soccer.jpg')

# blurred_image = cv2.GaussianBlur(image,(11,11),2)
# blur = cv2.blur(image,(5,5))

# edge_image = cv2.Canny(image,200,200)

# cv2.imshow(winname='edge_image',mat=edge_image)
# cv2.imshow(winname='blur',mat=blur)
# cv2.imshow(winname='blurred_image',mat=blurred_image)
# cv2.imshow(winname='frame',mat=image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Module 6
# ----------------------------------------------------------------
# Object Detection
# ----------------------------------------------------------------

from ultralytics import YOLO

# Load model
model = YOLO('yolo12.pt')
results = model(source=1,show = True)

# # Access the results
for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box
    print(names,confs,xywh,sep='\n')


