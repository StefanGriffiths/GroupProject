#TEST PROGRAM
# import the necessary packages
from sklearn.cluster import DBSCAN
import cv2
import imutils
#import numpy as np
import time
import pyrebase

config ={
  "apiKey": "AIzaSyDTfUqSEijy-UPrh6ab7VsRemlb87hr3tI",
  "authDomain": "groupproject-d2abd.firebaseapp.com",
  "databaseURL": "https://groupproject-d2abd.firebaseio.com",
  "storageBucket": "groupproject-d2abd.appspot.com"
}

firebase = pyrebase.initialize_app(config)

caps = cv2.VideoCapture(r'C:/Users/Emyr/Documents/Jupyter/pedestrian-detection/video/Ped4.MOV')
#gets frame count
countMax = int(caps.get(cv2.CAP_PROP_FRAME_COUNT))
count = 0
#tracks number of detections

#Cluster colours
#          Red       Yellow      Blue      Green     Purple      Orange
colors = [(255,0,0),(255,255,0),(0,0,255),(0,128,0),(128,0,128),(255,99,71)]

# YOLO weights and model configuration filepaths.
weightsPath = "C:\Users\Emyr\Documents\Jupyter\pedestrian-detection\yolo-coco\yolov3.weights"
configPath = "C:\Users\Emyr\Documents\Jupyter\pedestrian-detection\yolo-coco\yolov3.cfg"

# load YOLO trained on COCO dataset.
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#Will allow GPU processing in future.
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)  
#print(net)
# Get output layer names from YOLO
layerNames = net.getLayerNames()
layerNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#Detection Params
confiThresh = 0.5
overlapThresh = 0.2

while count < countMax - 1: #loops until all frames have been procesed.

        #start timer
        start = time.time()
        #reset variables
        center = []
        centerFull = []
        boxFull = []

        #read new frame
        ret, frame = caps.read()
        #resize frame, smaller image = less processing required.
        frame = imutils.resize(frame, width=min(frame.shape[0], frame.shape[1]))
        #dislay framecount on top left corner of frame.
        text = str(count)
        cv2.putText(frame, text, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        #get spatial dimensions of image
        (H, W) = frame.shape[:2]

        # construct a blob from the input image and then perform a forward pass
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416,416), swapRB=True, crop=False)
        
        #Input image data into yolo
        net.setInput(blob)

        #returns model data.
        layerOutputs = net.forward(layerNames)
        #reset arrays
        boxes = []
        confidences = []
        classIDs = []        

        #YOLO Output Processing
        for output in layerOutputs:
                # loop through each detections
                for detection in output:
                        #print(detection)
                        # extract class ID and confidence.
                        scores = detection[5:6]
                        #returns index of highest score
                        classID = np.argmax(scores)
                        confidence = scores[classID]
                        
                        # filter out weak predictions by ignoring low confidence detections
                        if confidence > confiThresh:

                                # scale the bounding box coords back relative to the size of the image.
                                box = detection[0:4] * np.array([W, H, W, H])
                        
                                (centerX, centerY, width, height) = box.astype("int")

                                # use the center (x, y)-coordinates to find the top and left corner of the bounding box

                                #Find corners of bounding box from centerpoints and width/height.
                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))

                                # update arrays of bounding box coords, confidences, centerpoints and class IDs

                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)
                                center.append([centerX, centerY])

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes, returns index positions
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confiThresh, overlapThresh)

        for i in idxs.flatten():
                #tranfers centerpoint datat to new array, excluding boundingboxes cut out by the NMS 
                centerFull.append(center[i])
                #extracts bounding box data of retained boxes.
                x, y = (boxes[i][0], boxes[i][1])
                w, h = (boxes[i][2], boxes[i][3])
                #tranfers bounding box data to new array, excluding boundingboxes cut out by the NMS (solves indexing issue) 
                boxFull.append([x,y,w,h])
                
        # ensure at least one detection exists
        if len(idxs) > 0:
            #cluster pedestrian centerpoints.
            db = DBSCAN(eps= 67, min_samples = 3, n_jobs = -1, leaf_size = 15).fit(centerFull)
            labels = db.labels_
            
            # Gets number of clusters
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # loop over the indexes we are keeping
            for i in range(len(boxFull)):
                # extract the bounding box coordinates
                x, y = (boxFull[i][0], boxFull[i][1])
                w, h = (boxFull[i][2], boxFull[i][3])
      
                cv2.circle(frame,(centerFull[i][0], centerFull[i][1]), 5, (0,255,0), -1)
                # draw a bounding box rectangle and label on the image
                if labels[i] == -1:
                                    
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
                        text = "{:.4f}".format(confidences[i])
                        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                else:
                                    
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (colors[labels[i]][0], colors[labels[i]][1], colors[labels[i]][2]), 2)
                        text = "{:.4f}".format(confidences[i])
                        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (colors[labels[i]][0], colors[labels[i]][1], colors[labels[i]][2]), 2)
                        
            
            #center = np.asarray(center)

        end = time.time()
        print("[INFO] Program took {:.2f} seconds.".format(end - start))
        cv2.imshow("Tracker", frame)
        cv2.waitKey(1)
        count +=1
        

caps.close()
cv2.destroyAllWindows()
            
            
        
