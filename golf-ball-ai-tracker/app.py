import redis
import torch
import numpy as np
import cv2
import subprocess as sp
from time import time
from collections import deque

class GolfBallDetector:
    def __init__(self, input, output, model_name):
        self.input = input
        self.output = output
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.r = redis.Redis(host="192.168.178.24")
        print("\n\nDevice Used:", self.device)

    def load_model(self, model_name):
        if model_name:
            model = torch.hub.load(
                "ultralytics/yolov5", "custom", path=model_name, force_reload=True
            )
        else:
            model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        return model

    def get_video(self):
        return cv2.VideoCapture(self.input)

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        x1, y1, x2, y2 = 0, 0, 0, 0

        highest_confidence = 0
        highest_confidence_index = 0

        for i in range(n):
            row = cord[i]
            if row[4] > highest_confidence:
                highest_confidence = row[4]
                highest_confidence_index = i

        if highest_confidence != 0:
            row = cord[highest_confidence_index]
            if row[4] >= 0.35:
                x1, y1, x2, y2 = (
                    int(row[0] * x_shape),
                    int(row[1] * y_shape),
                    int(row[2] * x_shape),
                    int(row[3] * y_shape),
                )
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(
                    frame,
                    self.class_to_label(labels[i]),
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    bgr,
                    2,
                )

        return frame, (x1, y1)

    def draw_trace(self, frame, points, prev_center, predicted):
        bgr = (0, 255, 0)
        transparent_img = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # loop over the set of tracked points
        for i in range(1, len(points)):
            # if either of the tracked points are None, ignore
            # them
            if points[i - 1] is None or points[i] is None:
                continue
            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(45 / float(i + 1)) * 3)
            cv2.line(transparent_img, points[i - 1], points[i], (0, 0, 255, 255), thickness)
        
        
        cv2.rectangle(transparent_img, (prev_center[1] - 208, prev_center[0] - 208), (prev_center[1] + 208, prev_center[0] + 208), bgr, 2)
        cv2.putText(transparent_img, "AI TRACER ENABLED", (725, 75), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,0,0), 3, cv2.LINE_AA)
        cv2.putText(transparent_img, "(experimental)", (840, 125), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,9), 2, cv2.LINE_AA)

        if predicted[0] < 0:
            predicted = (0, predicted[1])
        if predicted[1] < 0:
            predicted = (predicted[0], 0)
        if predicted[0] > 1920:
            predicted = (1920, predicted[1])
        if predicted[1] > 1080:
            predicted = (predicted[0], 1080)

        # bgr = (255, 0, 0)
        # cv2.circle(frame, predicted, 10, bgr, 2)

        return transparent_img
        

    def __call__(self):

        self.r.set("detection", 1)

        player = self.get_video()
        assert player.isOpened()
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Video information
        fps = 30
        width = 1920
        height = 1080
        rtmp_output = "rtmp://192.168.178.24:1935/live/local-final"

        # ffmpeg command
        # OpenCV does not support audio.
        command = ['ffmpeg',
                '-y',
                '-re', # '-re' is requiered when streaming in "real-time"
                '-f', 'rawvideo',
                '-thread_queue_size', '24576',  # May help https://stackoverflow.com/questions/61723571/correct-usage-of-thread-queue-size-in-ffmpeg
                # '-vcodec','rawvideo',
                '-pix_fmt', 'rgb24',
                '-s', "{}x{}".format(width, height),
                '-r', str(fps),
                # '-use_wallclock_as_timestamps', '1',
                '-i', '-',
                # '-use_wallclock_as_timestamps', '1',
                # '-itsoffset', '2',
                # '-fflags', 'nobuffer',
                '-i', self.input,
                "-filter_complex", "[0:v]colorkey=0x000000:0.5:0.0[fg];[1][fg]overlay",
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                # '-af', 'aresample=async=1',
                # '-adrift_threshold', '1',
                # '-async', '1',
                # '-vsync', '-1',
                '-c:a', 'aac',  # Select audio codec
                '-f', 'flv', 
                rtmp_output]

        
        # Pipeline configuration
        p = sp.Popen(command, stdin=sp.PIPE)

        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.output, four_cc, 30, (x_shape, y_shape))

        points = deque(maxlen=45)

        start_location = (872, 960)
        prev_center = (872, 960)

        frames_without_detection = 0
        predicted = (0, 0)

        while player.isOpened():
            start_time = time()
            ret, frame = player.read()
            if not ret:
                break

            if int(self.r.get("detection")):

                cropped_frame = frame[
                    prev_center[0] - 208 : prev_center[0] + 208,
                    prev_center[1] - 208 : prev_center[1] + 208,
                ]

                results = self.score_frame(cropped_frame)
                cropped_frame, new_center = self.plot_boxes(results, cropped_frame)

                if new_center != (0, 0):
                    prev_center = (
                        prev_center[0] - 208 + new_center[1] - 150,
                        prev_center[1] - 208 + new_center[0],
                    )
                    points.appendleft((prev_center[1], prev_center[0] + 150))
                    frames_without_detection = 0
                    # predicted = kf.predict(prev_center[1], prev_center[0] + 150)
                else:
                    frames_without_detection += 1
                    # predicted = kf.predict(predicted[0], predicted[1])
 
                if prev_center[0] < 208:
                    prev_center = (208, prev_center[1])
                if prev_center[1] < 208:
                    prev_center = (prev_center[0], 208)
                if prev_center[0] > 872:
                    prev_center = (872, prev_center[1])
                if prev_center[1] > 1712:
                    prev_center = (prev_center[0], 1712)

                # print(prev_center)

                # print(frames_without_detection)
                if frames_without_detection > 25:
                    prev_center = start_location
                    frames_without_detection = 0
                    points = deque(maxlen=45)
                    # predicted = (0, 0)

                finished_frame = self.draw_trace(frame, points, prev_center, predicted)

                end_time = time()
                fps = 1 / np.round(end_time - start_time, 3)

                # cv2.imwrite('C:\\images\\cat.jpeg', finished_frame)

                p.stdin.write(finished_frame.tobytes())
            
            else:
                transparent_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
                p.stdin.write(transparent_img.tobytes())

                
            # print(f"Frames Per Second : {fps}")
            # out.write(frame)

            # write to pipe
            

            # cv2.imshow("output", finished_frame)
             # Press Q on keyboard to  exit
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            transparent_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            # write to pipe
            p.stdin.write(transparent_img.tobytes())
        
        p.stdin.close()  # Close stdin pipe
        p.wait()


gbd = GolfBallDetector("rtmp://192.168.178.24:1935/live/local-test", "output.avi", "best-1.pt")
gbd()
