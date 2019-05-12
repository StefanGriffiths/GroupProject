#TEST PROGRAM
# import the necessary packages
from sklearn.cluster import DBSCAN
import cv2
import imutils
import numpy as np
import time
import os
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

while count < countMax - 1: #loops until all frames have been procesed.
        #reset variables
        start = time.time()
        center = []
        centerFull = []
        boxFull = []

        #read new frame
        ret, frame = caps.read()
        #resize frame, smaller image = less processing required.
        frame = imutils.resize(frame, width=min(frame.shape[0], frame.shape[1]))

        #dislay framecount on top left corner of frame.
        text = str(count)
        cv2.putText(frame, text, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # YOLO weights and model configuration filepaths.
        weightsPath = "C:\Users\Emyr\Documents\Jupyter\pedestrian-detection\yolo-coco\yolov3.weights"
        configPath = "C:\Users\Emyr\Documents\Jupyter\pedestrian-detection\yolo-coco\yolov3.cfg"

        # load YOLO trained on COCO dataset.
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        #get spatial dimensions of image
        (H, W) = frame.shape[:2]

        # Get output layer names from YOLO
        layerNames = net.getLayerNames()
        layerNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        # construct a blob from the input image and then perform a forward
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        #Input image data into yolo
        net.setInput(blob)

        #returns model data.
        layerOutputs = net.forward(layerNames)
        
        boxes = []
        confidences = []
        classIDs = []

        #PARAMETERS
        inpConfidence = 0.5
        inpThreshold = 0.4

        # loop through layer outputs
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
                        if confidence > inpConfidence:

                                # scale the bounding box coords back relative to the size of the image.
                                box = detection[0:4] * np.array([W, H, W, H])
                        
                                (centerX, centerY, width, height) = box.astype("int")

                                # use the center (x, y)-coordinates to derive the top and
                                # and left corner of the bounding box

                                #Find corners of boudning box from centerpoints and width/height.
                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))

                                # update our list of bounding box coordinates, confidences, centerpoints
                                # and class IDs

                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)
                                center.append([centerX, centerY])

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, inpConfidence, inpThreshold)
        
        for i in idxs.flatten():
                centerFull.append(center[i])
                
                x, y = (boxes[i][0], boxes[i][1])
                w, h = (boxes[i][2], boxes[i][3])

                boxFull.append([x,y,w,h])
                
        # ensure at least one detection exists
        if len(idxs) > 0:
            #cluster pedestrian centerpoints.
            db = DBSCAN(eps= 60, min_samples = 3, n_jobs = -1).fit(centerFull)
        
            labels = db.labels_
            
            # Gets number of clusters
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

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
                        
            
            center = np.asarray(center)

        #print('Estimated number of clusters: %d' % n_clusters_)
        #plt.show()
        end = time.time()
        print("[INFO] Program took {:.2f} seconds.".format(end - start))
        cv2.imshow("Tracker", frame)
        cv2.waitKey(1)
        count +=1
        

caps.close()
cv2.destroyAllWindows()
            
            
        
