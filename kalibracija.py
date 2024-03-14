# import the opencv library 
import cv2 
import numpy as np

HEIGHT = 8
WIDTH = 5
CELL_SIZE = 30 # mm

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
single_object_points = np.zeros((HEIGHT * WIDTH, 3), np.float32)
for i in range(HEIGHT):
    for j in range(WIDTH):
        single_object_points[i*WIDTH + j, :] = np.array(
                [j*CELL_SIZE, i*CELL_SIZE, 0], dtype=np.float32)


vid = cv2.VideoCapture(0) 

image_points = []
object_points = []

while(True): 
      
    ok, frame = vid.read() 
    
    if ok and cv2.waitKey(1) & 0xFF == ord('c'):
        # pretvorimo u grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_frame, (WIDTH, HEIGHT), None)

        if ret:
            corners2 = cv2.cornerSubPix(
                gray_frame, corners, (11, 11), (-1, -1), criteria)
            
            object_points.append(single_object_points)
            image_points.append(corners2)
            
            frame = cv2.drawChessboardCorners(
                frame, (WIDTH, HEIGHT), corners2, ret)
    cv2.imshow('img', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
   
vid.release() 
cv2.destroyAllWindows() 
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, gray_frame.shape[::-1], None, None)

mean_error = 0
for i in range(len(object_points)):
    imgpoints2, _ = cv2.projectPoints(single_object_points, rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print("total error: {}".format(mean_error/len(object_points)))

filename = "calib.yaml"
file = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
file.write("distortion", dist)
file.write("intrinsic", mtx)
file.release()