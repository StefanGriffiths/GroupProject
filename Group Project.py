#TEST PROGRAM - CUTTING DOWN
# import the necessary packages
from __future__ import print_function
#from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import imutils
import cv2
import sys
import matplotlib.pyplot as plt
from itertools import cycle
import random


class PeopleTracker:

    hog = cv2.HOGDescriptor()
    caps = cv2.VideoCapture(r'C:/Users/Emyr/Documents/Jupyter/pedestrian-detection/video/Ped4.MOV')
    count = int(caps.get(cv2.CAP_PROP_FRAME_COUNT))
    center = []
    recCount = 0
    pick = 0
    i = 0

    def BBoxes(self, frame):
        #frame = imutils.resize(frame, width = min(frame.shape[0], frame.shape[1]))
        frame = imutils.resize(frame, width= 1000,height = 1100)

        # detect people in the image
        (rects, weights) = self.hog.detectMultiScale(frame, winStride=(2,2), padding=(1, 1), scale=0.5)
        
        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        
        self.pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # draw the final bounding boxes
        self.recCount  = 0
        
        for (xA, yA, xB, yB) in self.pick:
          
            #cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            
            CentxPos = int((xA + xB)/2)
            CentyPos = int((yA + yB)/2)
        
            cv2.circle(frame,(CentxPos, CentyPos), 5, (0,255,0), -1)
            self.recCount += 1
            
            if len(rects) >1:
                   self.center.append([CentxPos, CentyPos])
          

        return frame


    def Clustering(self, frame):
        
        db = DBSCAN(eps= 70, min_samples = 2).fit(self.center)
        
        labels = db.labels_
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print("Labels: ", labels)
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        #print("Unique Labels: ", unique_labels)
        
        colors = plt.cm.rainbow(np.linspace(0, 255, len(unique_labels)))

        #colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for k in range(len(unique_labels)) ]
        
        print(colors)
        
        i = 0
        
        for (xA, yA, xB, yB) in self.pick:
            
            if labels[i] == -1:
                cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 0, 0), 2)
                i += 1
            else:
              
                #cv2.rectangle(frame, (xA, yA), (xB, yB), (r, g, b), 2)
                cv2.rectangle(frame, (xA, yA), (xB, yB), (colors[labels[i]][0] * 255, colors[labels[i]][1] * 255, colors[labels[i]][2] * 255), 2)
                i += 1
        
        
        #print("Colours: ", colors)
        center = np.asarray(self.center)
        
        #fig, ax = plt.subplots()
            
        #ax.set_xlim(0,frame.shape[1])
        #ax.set_ylim(frame.shape[0], 0)
        
        #for k, col in zip(unique_labels, colors):
            
            #if k == -1:
                 #Black used for noise.
                 #col = [0, 0, 0, 1]

            #class_member_mask = (labels == k)
            #xy = center[class_member_mask]
            #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=8)
        


def main():

    PT = PeopleTracker()
    PT.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    while PT.count > 1:

        PT.center = []

        ret, frame = PT.caps.read()

        frame = PT.BBoxes(frame)

        if PT.recCount >= 2:

            PT.Clustering(frame)
            

            #plt.title('Estimated number of clusters: %d' % n_clusters_)
            #plt.show()   
            cv2.imshow("Tracker", frame)
            cv2.waitKey(0)
            #cv2.destroyAllWindows()
            PT.count = PT.count - 1

        else:
            
            cv2.imshow("Tracker", frame)
            cv2.waitKey(0)
            #cv2.destroyAllWindows()
            PT.count = PT.count - 1

cv2.destroyAllWindows()


main()
