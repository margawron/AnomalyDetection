import cv2
import numpy as np
import json
from skimage.feature import hog
import time


def playVideo(filename):
    

    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (0, 25)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 255, 255)
    # Line thickness of 2 px
    thickness = 2
    
    cap = cv2.VideoCapture(filename)
    score_vid = np.load('score_vid.npy')
    counter = 0
    ret, frame = cap.read()
    
    while ret:
        if(counter < len(score_vid)):
            text = 'Normality score {:.2f}'.format(score_vid[counter]) 
            frameText = 'Frame number {:d}'.format(counter)
            if (score_vid[counter] > 0.95):
                color = (0,255,0)
            elif (score_vid[counter] > 0.80 and score_vid[counter] < 0.95):
                color = (0,255,255)
            else:
                color = (0,0,255) 
        
        frame = cv2.putText(frame, text, org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, frameText, (0,350), font, 
                        fontScale, (255,255,255), thickness, cv2.LINE_AA)
        cv2.imshow('Anomaly detection preview', frame)
        time.sleep(0.1)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        ret, frame = cap.read()
        counter+=1
    cap.release()
    cv2.destroyAllWindows()
