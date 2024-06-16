import numpy as np
import cv2 as cv
from cv2 import aruco
from math import (
    asin, pi, atan2, sqrt, cos
)

ARUCO_DICT = {
    "DICT_4x4_50": cv.aruco.DICT_4X4_50,
    "DICT_4x4_100": cv.aruco.DICT_4X4_100,
    "DICT_4x4_250": cv.aruco.DICT_4X4_250,
    "DICT_4x4_1000": cv.aruco.DICT_4X4_1000,
    "DICT_5x5_50": cv.aruco.DICT_5X5_50,
    "DICT_5x5_100": cv.aruco.DICT_5X5_100,
    "DICT_5x5_250": cv.aruco.DICT_5X5_250,
    "DICT_5x5_1000": cv.aruco.DICT_5X5_1000,
    "DICT_6x6_50": cv.aruco.DICT_6X6_50,
    "DICT_6x6_100": cv.aruco.DICT_6X6_100,
    "DICT_6x6_250": cv.aruco.DICT_6X6_250,
    "DICT_6x6_1000": cv.aruco.DICT_6X6_1000
}
    
BLUE = (255, 50, 50)
GREEN = (50, 255, 50)
RED = (50, 50, 255)
WHITE = (255, 255, 255)
PURPLE = (255, 50, 255)

def marker_display(corners, ids, rejected, image): 
    if len(corners) > 0:
        ids = ids.flatten()
        for markerCorner, markerID in zip(corners, ids):
                
                corners = markerCorner.reshape((4, 2))

                (topLeft, topRight, bottomRight, bottomLeft) = corners

                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                cv.line(image, topLeft, topRight, GREEN, 2)
                cv.line(image, topRight, bottomRight, GREEN, 2)
                cv.line(image, bottomRight, bottomLeft, GREEN, 2)
                cv.line(image, bottomLeft, topLeft, GREEN, 2)

                # compute and draw the center (x, y) - coordinates of the ArUco marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)

                cv.circle(image, (cX, cY), 4, PURPLE, -1)

                cv.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10),
                        cv.FONT_HERSHEY_PLAIN,
                        0.5, 
                        (0, 255, 0), 
                        2)
    return image

# roll -> rotation about x-axis (RED axis)
# pitch -> rotation about y-axis (GREEN axis)
# yaw -> rotation about z-axis (BLUE axis)
def print_coordinates(id, x, y, z, roll, pitch, yaw):
    print("---")
    print("marker i = ", id)
    print("coordinates:")
    print("  - position:")
    print('      - X:       {:.3f}'.format(x))
    print('      - Y:       {:.3f}'.format(y))
    print('      - H:       {:.3f}'.format(z))
    print("  - orientation:")
    print('      - roll:    {:.3f}'.format(roll))
    print('      - pitch:   {:.3f}'.format(pitch))
    print('      - yaw:     {:.3f}'.format(yaw))

def main():

    calibration_data_path = "calib_data/MultiMatrix.npz"
    calibration_data = np.load(calibration_data_path)
    cam_mat = calibration_data["camMatrix"]
    dist_coef = calibration_data["distCoef"]

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, parameters)

    MARKER_SIZE = 4 # centimeters

    while cap.isOpened():

        # Capture frame-by-frame.
        ret, img = cap.read()
        if not ret:
            break

        # Detecting markers.
        corners, ids, reject = detector.detectMarkers(img)

        # Display markers.
        img = marker_display(corners, ids, reject, img)

        if corners:
            # Getting rotation and traslation vectors.
            rVec, tVec, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, cam_mat, dist_coef)
            total_markers = range(0, ids.size)
            for ids, corners, i in zip(ids, corners, total_markers):
                
                # Drawing axisies for the pose of the marker
                _ = cv.drawFrameAxes(img, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)

                corners = corners.reshape(4,2)
                corners = corners.astype(int)
                top_right = corners[0].ravel()
                bottom_right = corners[2].ravel()

                cv.putText(img, f"id: {ids[0]} Dist: {round(tVec[i][0][2], 2)}", top_right, cv.FONT_HERSHEY_PLAIN, 1.3, PURPLE, 2)
                cv.putText(img, f"x: {round(tVec[i][0][0], 3)} y: {round(tVec[i][0][1], 3)}", bottom_right, cv.FONT_HERSHEY_PLAIN, 1.3, PURPLE, 2)
                
                x       = tVec[i][0][0]
                y       = tVec[i][0][1]
                z       = tVec[i][0][2]

                rotation_matrix, _ = cv.Rodrigues(rVec)
                angle_vec = calculateEulerAngle(rotation_matrix)
             
                print_coordinates(i, x, y, z, angle_vec[0], angle_vec[1], angle_vec[2])


        # Display the resulting frame with axies.
        cv.imshow('Main window', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


def calculateEulerAngle(rotation_matrix):

    R11 = rotation_matrix[0][0]
    R12 = rotation_matrix[0][1]
    R13 = rotation_matrix[0][2]
    R21 = rotation_matrix[1][0]
    R22 = rotation_matrix[1][1]
    R23 = rotation_matrix[1][2]
    R31 = rotation_matrix[2][0]
    R32 = rotation_matrix[2][1]
    R33 = rotation_matrix[2][2]

    if R31 != 1 and R31 != -1: 
        pitch_1 = -1*asin(R31)
        pitch_2 = pi - pitch_1 
        roll_1 = atan2( R32 / cos(pitch_1) , R33 /cos(pitch_1)) 
        roll_2 = atan2( R32 / cos(pitch_2) , R33 /cos(pitch_2)) 
        yaw_1 = atan2( R21 / cos(pitch_1) , R11 / cos(pitch_1))
        yaw_2 = atan2( R21 / cos(pitch_2) , R11 / cos(pitch_2)) 

        pitch = pitch_1 
        roll = roll_1
        yaw = yaw_1 
    else: 
        yaw = 0 # anything (we default this to zero)
        if R31 == -1: 
            pitch = pi/2 
            roll = yaw + atan2(R12,R13) 
        else: 
            pitch = -pi/2 
            roll = -1 * yaw + atan2(-1 * R12,-1 * R13) 

    # convert from radians to degrees
    roll = roll * 180/pi 
    pitch = pitch * 180/pi
    yaw = yaw * 180/pi 
    rxyz_deg = [roll , pitch , yaw]

    return rxyz_deg

def calculateEulerAngleSimply(rotation_matrix):
    R11 = rotation_matrix[0][0]
    R12 = rotation_matrix[0][1]
    R13 = rotation_matrix[0][2]
    R21 = rotation_matrix[1][0]
    R22 = rotation_matrix[1][1]
    R23 = rotation_matrix[1][2]
    R31 = rotation_matrix[2][0]
    R32 = rotation_matrix[2][1]
    R33 = rotation_matrix[2][2]

    roll = atan2(R32, R33)
    pitch = atan2(-R31, sqrt(R32*R32+ R33*R33))
    yaw = atan2(R21, R11)

    # convert from radians to degrees
    roll = roll * 180/pi 
    pitch = pitch * 180/pi
    yaw = yaw * 180/pi 
    rxyz_deg = [roll , pitch , yaw]

    return rxyz_deg


if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    main()
    cap.release()
    cv.destroyAllWindows()