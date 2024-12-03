import numpy as np
import cv2 as cv
from cv2 import aruco
import urllib.request
from math import (
    asin, pi, atan2, sqrt, cos
)
import websocket


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

# =====================================================================


def main():

    calibration_data_path = "calib_data/MultiMatrix.npz"
    calibration_data = np.load(calibration_data_path)
    cam_matrix = calibration_data["camMatrix"]
    dist_coef = calibration_data["distCoef"]

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, parameters)

    MARKER_SIZE = 4  # centimeters

    while True:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)

        im = cv.imdecode(imgnp, -1)

        # Detecting markers.
        corners, ids, reject = detector.detectMarkers(im)
        
        # Display markers.
        im = marker_display(corners, ids, reject, im)

        if corners:
            # Getting rotation and traslation vectors.
            # rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camMatrix, distCoef)
            rvec, tvec, _ = my_estimatePoseSingleMarkers(corners, MARKER_SIZE, cam_matrix, dist_coef)
            total_markers = range(0, ids.size)
            for ids, corners, i in zip(ids, corners, total_markers):
                # Drawing axisies for the pose of the marker
                _ = cv.drawFrameAxes(im, cam_matrix, dist_coef, rvec[i], tvec[i], 4, 4)

                corners = corners.reshape(4, 2)
                corners = corners.astype(int)
                top_right = corners[0].ravel()
                bottom_right = corners[2].ravel()

                cv.putText(im, f"id: {ids[0]} Dist: {tvec[i][0]}", top_right, cv.FONT_HERSHEY_PLAIN, 1.3,
                           PURPLE, 2)
                cv.putText(im, f"x: {tvec[i][1]} y: {tvec[i][2]}", bottom_right,
                           cv.FONT_HERSHEY_PLAIN, 1.3, PURPLE, 2)
                

                x = tvec[i][2][0]
                y = tvec[i][1][0]
                z = tvec[i][0][0]

                rotation_matrix, _ = cv.Rodrigues(rvec[i])
                angle_vec = calculateEulerAngle(rotation_matrix)
                print_coordinates(i, x, y, z, angle_vec[0], angle_vec[1], angle_vec[2])

                my_msg = str(x) + "," + str(y) + "," + str(z) + "," + str(angle_vec[0]) + "," + str(angle_vec[1]) + "," + str(angle_vec[2]) + '\n'
    
                ws.send(my_msg)
                # Wait for server to respond and print it
                result = ws.recv()
                print("Received: " + result)

                # -------------------------------

        cv.imshow('live cam testing', im)
        key = cv.waitKey(5)
        if key==ord('q'):
            break

# ======================================================================

# КРЕН roll -> rotation about x-axis (RED axis)
# ТАНГАЖ pitch -> rotation about y-axis (GREEN axis)
# КУРС yaw -> rotation about z-axis (BLUE axis)
def print_coordinates(id, x, y, z, roll, pitch, yaw):
    print("---")
    print("marker i = ", id)
    print("coordinates:")
    print("  - position:")
    print('- X:       ', x)
    print('- Y:       ', y)
    print('- H:       ', z)
    print("  - orientation:")
    print('      - roll:    ', roll)
    print('      - pitch:   ', pitch)
    print('      - yaw:     ', yaw)

# ======================================================================

def calculateEulerAngle(rotation_matrix):
    r11 = rotation_matrix[0][0]
    r12 = rotation_matrix[0][1]
    r13 = rotation_matrix[0][2]
    r21 = rotation_matrix[1][0]
    r22 = rotation_matrix[1][1]
    r23 = rotation_matrix[1][2]
    r31 = rotation_matrix[2][0]
    r32 = rotation_matrix[2][1]
    r33 = rotation_matrix[2][2]

    if r31 != 1 and r31 != -1:
        pitch_1 = -1 * asin(r31)
        pitch_2 = pi - pitch_1
        roll_1 = atan2(r32 / cos(pitch_1), r33 / cos(pitch_1))
        roll_2 = atan2(r32 / cos(pitch_2), r33 / cos(pitch_2))
        yaw_1 = atan2(r21 / cos(pitch_1), r11 / cos(pitch_1))
        yaw_2 = atan2(r21 / cos(pitch_2), r11 / cos(pitch_2))

        pitch = pitch_1
        roll = roll_1
        yaw = yaw_1
    else:
        yaw = 0  # anything (we default this to zero)
        if r31 == -1:
            pitch = pi / 2
            roll = yaw + atan2(r12, r13)
        else:
            pitch = -pi / 2
            roll = -1 * yaw + atan2(-1 * r12, -1 * r13)

            # convert from radians to degrees
    roll = roll * 180 / pi
    pitch = pitch * 180 / pi
    yaw = yaw * 180 / pi
    rxyz_deg = [roll, pitch, yaw]

    return rxyz_deg

# ======================================================================

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        success, r, t = cv.solvePnP(marker_points, c, mtx, distortion, flags=cv.SOLVEPNP_ITERATIVE)
        rvecs.append(r)
        tvecs.append(t)
        trash.append(success)
    return rvecs, tvecs, trash

# ======================================================================

if __name__ == '__main__':

    # Camera initialize
    url = 'http://192.168.1.10/cam-lo.jpg'
    cv.namedWindow("live cam testing", cv.WINDOW_AUTOSIZE)
    cap = cv.VideoCapture(url)
    # cap = cv.VideoCapture(0, cv.CAP_DSHOW)

    if not cap.isOpened():
        print("Failed to open camera")
        exit()

    # websocket initialize
    ws = websocket.WebSocket()
    ws.connect("ws://192.168.1.13")
    print("Connected to WebSocket server")

    main()

    ws.close()
    cap.release()
    cv.destroyAllWindows()
