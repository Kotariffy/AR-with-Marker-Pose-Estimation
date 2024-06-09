import numpy as np
import cv2 as cv
from cv2 import aruco


BLUE = (255, 50, 50)
GREEN = (50, 255, 50)
RED = (50, 50, 255)
WHITE = (255, 255, 255)

def put_text(img, text, pos, color):
    return cv.putText(img, text, pos,
                       fontFace=cv.FONT_HERSHEY_DUPLEX,
                       fontScale=0.6, color=color)

def main():
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)

    parameters = aruco.DetectorParameters()

    detector = cv.aruco.ArucoDetector(dictionary, parameters)

    calibration_data_path = "calib_data/MultiMatrix.npz"

    calibration_data = np.load(calibration_data_path)

    cam_mat = calibration_data["camMatrix"]
    dist_coef = calibration_data["distCoef"]
    r_vector = calibration_data["rVector"]
    t_vector = calibration_data["tVector"]

    MARKER_SIZE = 4 # centimeters

    while True:

        # Capture frame-by-frame
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        markerCorners, markerIds, reject = detector.detectMarkers(frame)

        if markerCorners:
            rVec, tVec, _ = aruco.estimatePoseSingleMarkers(markerCorners, MARKER_SIZE, cam_mat, dist_coef)

            total_markers = range(0, markerIds.size)
            for ids, corners, i in zip(markerIds, markerCorners, total_markers):
                
                cv.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA)
                corners = corners.reshape(4,2)

                corners = corners.astype(int)
                top_right = corners[0].ravel()
                top_left = corners[1].ravel()
                bottom_right = corners[2].ravel()
                bottom_left = corners[3].ravel()

                # Draw the pose of the marker

                poit = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
                cv.putText(frame, 
                           f"id: {ids[0]}", 
                           top_right, 
                           cv.FONT_HERSHEY_PLAIN, 
                           1.3, 
                           (0, 255, 0), 
                           2, 
                           cv.LINE_AA)
        # Display the resulting frame
        cv.imshow('Main window', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture

if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    main()
    cap.release()
    cv.destroyAllWindows()