#TEST PROGRAM
# import the necessary packages
from sklearn.cluster import DBSCAN
import imutils
import numpy as np
import cv2
import time
import msvcrt
import os
import pyrebase

config =
{
  "apiKey": "AIzaSyDTfUqSEijy-UPrh6ab7VsRemlb87hr3tI",
  "authDomain": "groupproject-d2abd.firebaseapp.com",
  "databaseURL": "https://groupproject-d2abd.firebaseio.com",
  "storageBucket": "groupproject-d2abd.appspot.com"
}

firebase = pyrebase.initialize_app(config)

caps = cv2.VideoCapture(r'C:/Users/Emyr/Documents/Jupyter/pedestrian-detection/video/Ped4.MOV')
count = int(caps.get(cv2.CAP_PROP_FRAME_COUNT))

recCount = 0

#          Red       Yellow      Blue      Green     Purple      Orange
colors = [(255,0,0),(255,255,0),(0,0,255),(0,128,0),(128,0,128),(255,99,71)]

while count > 2:

        
        KeyboardInterrupt
        recCount = 0
        start = time.time()
        center = []

        ret, frame = caps.read()
        
        frame = imutils.resize(frame, width=min(frame.shape[0], frame.shape[1]))

        text = str(count)
        cv2.putText(frame, text, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        #print(frame.shape[0])
        #print(frame.shape[1])
        labelsPath = "C:\Users\Emyr\Documents\Jupyter\pedestrian-detection\yolo-coco\coco.names"
        LABELS = open(labelsPath).read().strip().split("\n")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = "C:\Users\Emyr\Documents\Jupyter\pedestrian-detection\yolo-coco\yolov3.weights"
        configPath = "C:\Users\Emyr\Documents\Jupyter\pedestrian-detection\yolo-coco\yolov3.cfg"

        #PARAMETERS
        inpConfidence = 0.7
        inpThreshold = 0.5

        # load our YOLO object detector trained on COCO dataset (80 classes)
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        
        (H, W) = frame.shape[:2]

        layerNames = net.getLayerNames()
        layerNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        
        layerOutputs = net.forward(layerNames)
        
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                        # extract the class ID and confidence (i.e., probability) of
                        # the current object detection
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > inpConfidence:
                                # scale the bounding box coordinates back relative to the
                                # size of the image, keeping in mind that YOLO actually
                                # returns the center (x, y)-coordinates of the bounding
                                # box followed by the boxes' width and height
                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")

                                # use the center (x, y)-coordinates to derive the top and
                                # and left corner of the bounding box
                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))

                                # update our list of bounding box coordinates, confidences, centerpoints
                                # and class IDs

                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)
                                
                                print("X", centerX)
                                print("Y", centerY)
                                
                                center.append([centerX, centerY])
                                    
                                cv2.circle(frame,(centerX, centerY), 5, (0,255,0), -1)
                                recCount += 1
            
                                      

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, inpConfidence, inpThreshold)

        if recCount >= 2:

            db = DBSCAN(eps= 65, min_samples = 3, n_jobs = -1).fit(center)
        
            labels = db.labels_
            
            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            #print("Labels: ", labels)
            # Black removed and is used for noise instead.
            unique_labels = set(labels)

            # ensure at least one detection exists
            if len(idxs) > 0:
                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                            # extract the bounding box coordinates
                            x, y = (boxes[i][0], boxes[i][1])
                            w, h = (boxes[i][2], boxes[i][3])
                            
                            # draw a bounding box rectangle and label on the image
                            if labels[i] == -1:
                                #color = [int(c) for c in self.COLORS[self.classIDs[i]]]
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
                                text = "{:.4f}".format(confidences[i])
                                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                            else:
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (colors[labels[i]][0], colors[labels[i]][1], colors[labels[i]][2]), 2)
                                text = "{:.4f}".format(confidences[i])
                                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (colors[labels[i]][0], colors[labels[i]][1], colors[labels[i]][2]), 2)
                        
            
            center = np.asarray(center)

        print('Estimated number of clusters: %d' % n_clusters_)
        #plt.show()
        end = time.time()
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))
        cv2.imshow("Tracker", frame)
        cv2.waitKey(1)
        count = count - 1
        

caps.close()
cv2.destroyAllWindows()
            
            
        
