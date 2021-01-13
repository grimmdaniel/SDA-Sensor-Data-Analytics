import numpy as np
import cv2
import sys

file_path = sys.argv[1]
trajectory_lines_needed = bool(int(sys.argv[2]))
cap = cv2.VideoCapture(file_path)
output_file_path = (file_path.split('\\')[1]).split('.')[0] + '_result.avi'


# params for ShiTomasi corner detection
feature_params = dict(maxCorners = 100,
                      qualityLevel = 0.08,
                      minDistance = 7,
                      blockSize = 3)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize = (15,15),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file_path, fourcc, 30, (old_frame.shape[1], old_frame.shape[0]), True)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(True):
    
    ret,frame = cap.read()
    
    if frame is None:
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        
        frame = cv2.circle(frame,(a,b),5,[0,0,255],-1)
        
        if trajectory_lines_needed:
            mask = cv2.line(mask, (a,b),(c,d), [0,255,0], 2)
            img = cv2.add(frame,mask)
        else:
            img = frame
        
    out.write(img)
    
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    
cap.release()
out.release()

print('File has been exported into: ' + output_file_path)