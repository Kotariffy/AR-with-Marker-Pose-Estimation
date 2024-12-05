import numpy as np
import cv2 as cv
from cv2 import aruco
from math import (
    asin, pi, atan2, sqrt, cos
)
import urllib.request
import websocket
from scipy.spatial.transform import Rotation

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

            cv.line(image, topLeft, topRight, BLUE, 2)
            cv.line(image, topRight, bottomRight, BLUE, 2)
            cv.line(image, bottomRight, bottomLeft, BLUE, 2)
            cv.line(image, bottomLeft, topLeft, BLUE, 2)

            # compute and draw the center (x, y) - coordinates of the ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)

            cv.circle(image, (cX, cY), 4, PURPLE, -1)

            cv.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10),
                       cv.FONT_HERSHEY_PLAIN,
                       0.5,
                       (0, 255, 0),
                       1)
    return image


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


def main():
    calibration_data_path = "calib_data/MultiMatrix.npz"
    calibration_data = np.load(calibration_data_path)
    cam_matrix = calibration_data["camMatrix"]
    dist_coef = calibration_data["distCoef"]

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, parameters)

    MARKER_SIZE = 4  # centimeters

    t = 0
    t, x, y, z, x_angle_deg, y_angle_deg, z_angle_deg, wx, wy, wz, w_angle_x, w_angle_y, w_angle_z = initialize()
    

    while True:

        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)

        img = cv.imdecode(imgnp, -1)
    
        # Detecting markers.
        corners, ids, reject = detector.detectMarkers(img)

        # Display markers.
        # img = marker_display(corners, ids, reject, img)

        objPoints = np.array([[-MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
                              [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
                              [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
                              [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]], dtype=np.float32)
        
        if ids is not None:
            for i in range(len(corners)):
                t = t + 1 
                if t == 1:
                    x_angle_deg_temp = 0
                    y_angle_deg_temp = 0
                    z_angle_deg_temp = 0
                    x_temp = 0
                    y_temp = 0
                    z_temp = 0
                
                else:
                    x_angle_deg_temp = x_angle_deg
                    y_angle_deg_temp = y_angle_deg
                    z_angle_deg_temp = z_angle_deg
                    x_temp = x
                    y_temp = y
                    z_temp = z

                
                success, rvec, tvec = cv.solvePnP(objPoints, corners[i], cam_matrix, dist_coef, flags=cv.SOLVEPNP_ITERATIVE)

                if success:
                    print(f'Шаг: {t}')
                    cv.drawFrameAxes(img, cam_matrix, dist_coef, rvec, tvec, MARKER_SIZE * 1.5, 2)
                    print(f'Расстояние:  {tvec.flatten()} мм')
                    x = tvec[0][0]
                    y = tvec[1][0]
                    z = tvec[2][0]

                    rotation_matrix, _ = cv.Rodrigues(rvec)
                    r = Rotation.from_matrix(rotation_matrix)
                    euler_angles = r.as_euler('ZYX', degrees=True)
                    z_angle_deg, y_angle_deg, x_angle_deg = euler_angles

                    if t > 1:
                        w_angle_x = x_angle_deg - x_angle_deg_temp
                        w_angle_y = y_angle_deg - y_angle_deg_temp
                        w_angle_z = z_angle_deg - z_angle_deg_temp 
                        wx = x - x_temp
                        wy = y - y_temp
                        wz = z - z_temp

                    print(f"Скорости поворота: wx={wx}, wy={wy}, wz={wz}", end='\n')
                    print(f"Скорости поворота углов: {w_angle_x}, {w_angle_y}, {w_angle_z}", end='\n')
                    print(f"Углы повороты: x={x_angle_deg}, y={y_angle_deg}, z={z_angle_deg}", end='\n\n')
                    
                    # Send to esp
                    my_msg = str(x) + "," + str(y) + "," + str(z) + "," + str(wx) + "," + str(wy) + "," + str(wz) + "," + str(x_angle_deg) + "," + str(y_angle_deg) + "," + str(z_angle_deg) + "," + str(w_angle_x) + "," + str(w_angle_y) + "," + str(w_angle_z) + '\n'
                    ws.send(my_msg)
                    # Wait for server to respond and print it
                    result = ws.recv()
                    print("Received: " + result)
                    

            aruco.drawDetectedMarkers(img, corners, ids, GREEN)
            
        else:
            t = 0
        # -------------------------------

        # Display the resulting frame with axies.
        cv.imshow('live cam testing', img)
        key = cv.waitKey(5)
        if key==ord('q'):
            break

    
def initialize():
    t = 0
    x = 0
    y = 0
    z = 0
    y_angle_deg = 0
    x_angle_deg = 0
    z_angle_deg = 0
    wx = 0
    wy = 0
    wz = 0
    w_angle_x = 0
    w_angle_y = 0
    w_angle_z = 0

    return t, x, y, z, x_angle_deg, y_angle_deg, z_angle_deg, wx, wy, wz, w_angle_x, w_angle_y, w_angle_z

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