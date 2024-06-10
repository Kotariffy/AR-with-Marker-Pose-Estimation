import numpy as np
import cv2 as cv
from cv2 import aruco

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

                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)

                cv.circle(image, (cX, cY), 4, BLUE, 2)

                cv.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10),
                        cv.FONT_HERSHEY_PLAIN,
                        0.5, 
                        (0, 255, 0), 
                        2)
    return image

def print_coordinates(id, x, y, z, angleX, angleY, angleZ):
    print("---")
    print("marker i = ", id)
    print("coordinates:")
    print("  - position:")
    print('      - X: {:.3f}'.format(x))
    print('      - Y: {:.3f}'.format(y))
    print('      - H: {:.3f}'.format(z))
    print("  - orientation:")
    print('      - wX: {:.3f}'.format(angleX * 57.2958))
    print('      - wY: {:.3f}'.format(angleY * 57.2958))
    print('      - wZ: {:.3f}'.format(angleZ * 57.2958))


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
                
                # Drawing the pose of the marker
                _ = cv.drawFrameAxes(img, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)

                corners = corners.reshape(4,2)
                corners = corners.astype(int)
                top_right = corners[0].ravel()
                bottom_right = corners[2].ravel()

                cv.putText(img, f"id: {ids[0]} Dist: {round(tVec[i][0][2], 2)}", top_right, cv.FONT_HERSHEY_PLAIN, 1.3, (200, 100, 0), 2)
                cv.putText(img, f"x: {round(tVec[i][0][0], 1)} y: {round(tVec[i][0][1], 1)}", bottom_right, cv.FONT_HERSHEY_PLAIN, 1.3, (200, 100, 0), 2)
                
                x       = tVec[i][0][0]
                y       = tVec[i][0][1]
                z       = tVec[i][0][2]
                angleX  = rVec[i][0][0] 
                angleY  = rVec[i][0][1] 
                angleZ  = rVec[i][0][2] 

                print_coordinates(i, x, y, z, angleX, angleY, angleZ)


        # Display the resulting frame with axies.
        cv.imshow('Main window', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    main()
    cap.release()
    cv.destroyAllWindows()