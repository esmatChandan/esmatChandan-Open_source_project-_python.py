import cv2 as cv
import mediapipe as mp
import time
import utils, math
import numpy as np
import pygame

#loading alarm audio
pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")

# variables initialization
frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0
IS_PLAYING = False

# constants
CLOSED_EYES_FRAME = 40
CLOSED_EYES_RATIO = 3.7 #3.7
FONTS = cv.FONT_HERSHEY_COMPLEX

# indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

NOSE = [168, 6, 197, 195, 5, 4, 1]

MOUTH = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]

map_face_mesh = mp.solutions.face_mesh
# camera object 
camera = cv.VideoCapture(0)


# landmark detection function
def landmarksDetection(img, results):
    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]

    # returning the list of tuples for each landmark
    return mesh_coord

def faceDirection(landmarks):
    pass

# distance
def distance(point1, point2):
    x, y = point1
    x1, y1 = point2
    d = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return d


# Blinking Ratio
def blinkRatio(landmarks, right_indices, left_indices):
    # Right eyes
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = distance(rh_right, rh_left)
    rvDistance = distance(rv_top, rv_bottom)

    lvDistance = distance(lv_top, lv_bottom)
    lhDistance = distance(lh_right, lh_left)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    eye_ratio = (reRatio + leRatio) / 2
    return round(eye_ratio, 2)


with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    # starting time here
    start_time = time.time()
    # starting Video loop.
    while True:
        frame_counter += 1  # frame counter
        ret, frame = camera.read()  # getting frame from camera
        if not ret:
            break

        frame = cv.flip(frame, 2) #flipping the frame
        #  resizing frame
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results)
            ratio = blinkRatio(mesh_coords, RIGHT_EYE, LEFT_EYE)
            utils.colorBackgroundText(frame, f'Ratio : {round(ratio, 2)}', FONTS, 0.7, (30, 100), 2, utils.PINK,
                                      utils.YELLOW)
            if ratio > CLOSED_EYES_RATIO:
                CEF_COUNTER += 1
            else:
                CEF_COUNTER = 0
                IS_PLAYING = False
                pygame.mixer.music.stop()
            if CEF_COUNTER > CLOSED_EYES_FRAME:
                if not IS_PLAYING:
                    pygame.mixer.music.play(-1) # -1 for infinite loop
                utils.colorBackgroundText(frame, f'Sleepy', FONTS, 1.7, (int(frame_height / 2), 100), 2, utils.YELLOW,
                                      pad_x=6, pad_y=6)
                IS_PLAYING = True

            #drawing the lines
            cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, utils.GREEN, 1,
                         cv.LINE_AA)
            cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, utils.GREEN, 1,
                         cv.LINE_AA)
            cv.polylines(frame, [np.array([mesh_coords[p] for p in MOUTH], dtype=np.int32)], True, utils.GREEN, 1,
                         cv.LINE_AA)
            cv.polylines(frame, [np.array([mesh_coords[p] for p in NOSE], dtype=np.int32)], False, utils.GREEN, 1,
                         cv.LINE_AA)
        else:
            utils.colorBackgroundText(frame, f'No Driver', FONTS, 1.7, (int(frame_height / 2), 100), 2, utils.YELLOW,
                                      pad_x=6, pad_y=6)

        # calculating  frame per seconds FPS
        end_time = time.time() - start_time
        fps = frame_counter / end_time

        frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)}', FONTS, 1.0, (30, 50), bgOpacity=0.5,
                                         textThickness=2)

        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        if key == ord('q') or key == ord('Q'):
            break
    cv.destroyAllWindows()
    camera.release()