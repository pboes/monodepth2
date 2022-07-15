
from depthPredictor import DepthPredictor
import cv2
import dlib
import time
import threading
import math
from PIL import Image
import sys
import numpy as np
from numpy.linalg import norm

carCascade = cv2.CascadeClassifier('myhaar.xml')
video = cv2.VideoCapture('videos/new york traffic.mp4')

WIDTH = 1280
HEIGHT = 720


def d2tod3(x, y, d, c, f):
    v = np.array([x, y])
    w = (v - c) / f * d
    return np.append(w, [d])


def estimate_speed_from_poisson(samples, fps):
    total_number_passing_cars = sum(samples)
    estimated_mean = total_number_passing_cars/len(samples)
    estimated_speed_in_frames = estimated_mean
    # TODO: Finish thinking about this
    estimated_speed = estimated_speed_in_frames * fps
    return estimated_speed


def estimateSpeed(
    location1,
    location2,
    depths,
    fps,
    use_depth_estimation=True
):
    if use_depth_estimation:
        x1, y1, w1, h1 = location1
        x2, y2, w2, h2 = location2
        f_pinhole = 100
        c = [WIDTH//2, HEIGHT//2]
        p1 = d2tod3(y1, x1, depths[y1, x1], c, f_pinhole)
        p2 = d2tod3(y2, x2, depths[y2, x2], c, f_pinhole)
        diff = p2 - p1
        d_meters = norm(diff)
        direction = -np.sign(depths[y2, x2] - depths[y1, x1])
    else:
        d_pixels = math.sqrt(math.pow(
            location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
        ppm = 8.8
        d_meters = d_pixels / ppm
    speed = direction * d_meters * fps * 3.6
    return speed


def trackMultipleObjects():
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    fps = video.get(cv2.CAP_PROP_FPS)

    carTracker = {}
    carNumbers = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    mov_queue = []
    mov_window_size = 7
    moving_average = 0

    tracker_counter = 0
    poisson_window = 300
    poisson_counter = []

    # Write output to video file
    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 10, (WIDTH, HEIGHT))

    _, first_frame = video.read()
    first_frame = cv2.resize(first_frame, (WIDTH, HEIGHT))
    print(first_frame.shape)
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    first_frame_pil = Image.fromarray(first_frame)
    depth_predictor = DepthPredictor()
    depths = depth_predictor.predict_depths(first_frame_pil)[0, 0]
    depth_predictor.save_depths_to_jpeg(depths, "depth_preds.jpeg")
    print(f"depths shape: {depths.shape}")
    np.save("depthmap.jpg", depths)

    while True:
        rc, image = video.read()
        if type(image) == type(None):
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()

        frameCounter = frameCounter + 1

        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)

            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            print('Removing carID ' + str(carID) + ' from list of trackers.')
            print('Removing carID ' + str(carID) + ' previous location.')
            print('Removing carID ' + str(carID) + ' current location.')
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        if frameCounter % poisson_window == 0:
            image_pil = Image.fromarray(image)
            depths = depth_predictor.predict_depths(first_frame_pil)[0, 0]

            poisson_counter.append(tracker_counter)
            poisson_estimate = estimate_speed_from_poisson(
                poisson_counter, fps)
            tracker_counter = 0

        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                matchCarID = None

                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()

                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())

                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                        matchCarID = carID

                if matchCarID is None:
                    print('Creating new tracker ' + str(currentCarID))
                    tracker_counter += 1
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(
                        image, dlib.rectangle(x, y, x + w, y + h))

                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]

                    currentCarID = currentCarID + 1

            # we consider the number of new car trackers generated in the last x frames:

        # cv2.line(resultImage,(0,480),(1280,480),(255,0,0),5)

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()

            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())

            cv2.rectangle(resultImage, (t_x, t_y),
                          (t_x + t_w, t_y + t_h), rectangleColor, 4)

            # speed estimation
            carLocation2[carID] = [t_x, t_y, t_w, t_h]

        # if not (end_time == start_time):
        #     fps = 1.0/(end_time - start_time)

        # cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (620, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        for i in carLocation1.keys():
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]

                # print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
                carLocation1[i] = [x2, y2, w2, h2]

                # print 'new previous location: ' + str(carLocation1[i])
                if [x1, y1, w1, h1] != [x2, y2, w2, h2] and y1 > HEIGHT*1/2:
                    if (speed[i] == None or speed[i] == 0) and y1 >= HEIGHT/2 and (y1 <= 3*HEIGHT/4 and x1 >= WIDTH/4) and x1 <= 3*WIDTH/4:
                        estimated_speed = estimateSpeed(
                            [x1, y1, w1, h1], [x2, y2, w2, h2], depths, fps)
                        speed[i] = estimated_speed
                        mov_queue.append(estimated_speed)
                        moving_average = int(sum(mov_queue)/len(mov_queue))
                        if len(mov_queue) >= mov_window_size:
                            mov_queue.pop()

                    # if y1 > 275 and y1 < 285:
                    if speed[i] != None:
                        cv2.putText(resultImage, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(
                            y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                    # print ('CarID ' + str(i) + ': speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')

                    # else:
                    #	cv2.putText(resultImage, "Far Object", (int(x1 + w1/2), int(y1)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        # print ('CarID ' + str(i) + ' Location1: ' + str(carLocation1[i]) + ' Location2: ' + str(carLocation2[i]) + ' speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')

        cv2.putText(
            resultImage,
            f"current traffic flow estimate: {moving_average} km/h",
            (10, HEIGHT - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        circle_radius = 50
        margin = 10
        #colormap = cm.get_cmap('inferno', 256)
        #cmap = np.linspace(0, 1, 256, endpoint=True)
        #cmap = colormap.to_rgba(cmap, bytes=True)

        cv2.circle(
            resultImage,
            ((circle_radius + margin), circle_radius + margin),
            circle_radius,
            (0, abs(moving_average), min(255, 255 - 2*abs(moving_average))),
            -1
        )

        # cv2.putText(
        #     resultImage,
        #     f"current Poisson traffic flow estimate: {poisson_estimate} km/h",
        #     (10, HEIGHT - 50),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)

        cv2.imshow('result', resultImage)
        # Write the frame into the file 'output.avi'
        # out.write(resultImage)

        if cv2.waitKey(33) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    trackMultipleObjects()
