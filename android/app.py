import cv2
import numpy as np
import xml.etree.ElementTree as ET

from joblib import load
from kivy.app import App
from kivy.clock import Clock
from plyer import filechooser
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from tensorflow.lite.python.interpreter import Interpreter


__version__ = "1.0.3"


def get_id(x, y, imW=1920):
    gs = 400  # grid_width
    return int((y//(gs)) * (imW//gs) + (x//gs))


def get_lane(x, y):
    if x < 590:
        return 1
    if x < 940 + 0.37 * y:
        return 2
    if x < 1320 + 0.8 * y:
        return 3
    return 4


# Parse XML file
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    vehicles = {}
    for vehicle in root.findall('./gtruth/vehicle'):
        if vehicle.get('radar') == "True":
            iframe = int(vehicle.get('iframe'))
            region = vehicle.find('region')
            x = int(region.get('x'))
            y = int(region.get('y'))
            w = int(region.get('w'))
            h = int(region.get('h'))

            radar = vehicle.find('radar')
            speed = float(radar.get('speed'))
            lane = int(vehicle.get('lane'))
            vehicles[iframe+1] = ((x, y, w, h), speed, lane)
    return vehicles


def load_tflite_model(path):
    interpreter = Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


class VideoApp(App):
    def build(self):
        self.orientation = "vertical"

        self.image = Image()
        # Add the image layout
        self.image_layout = BoxLayout(orientation='vertical')
        self.image_layout.add_widget(self.image)

        self.browse_btn = Button(text="Browse")
        self.browse_btn.size_hint_y = None  # Set height to fixed value
        self.browse_btn.height = 50  # Set fixed height
        self.browse_btn.bind(on_press=self.browse)

        self.image_layout.add_widget(self.browse_btn)

        self.layout = BoxLayout(orientation='vertical')

        self.layout.add_widget(self.image_layout)
        # self.layout.add_widget(self.image)
        # self.layout.add_widget(self.browse_btn)

        return self.layout

    def init(self):
        self.xml_path = '../data/vehicles.xml'
        # self.in_video_path = '../data/video.h264'
        self.out_video_path = '../data/out.mp4'
        self.license_plate_detector_model_path = '' +\
            '../models/license_plate_detector_float32.tflite'
        self.speed_prediction_model_path = '../models/speed_prediction_model.tflite'

        self.vehicles = parse_xml(self.xml_path)
        print('xml file was parsed successfully!\n')

        self.cap = cv2.VideoCapture(self.in_video_path)
        self.max_frames = 500
        if not self.max_frames:
            self.max_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(
            self.out_video_path, fourcc, fps, (width, height)
        )

        self.pi, self.pid, self.pod = load_tflite_model(
            self.license_plate_detector_model_path
        )
        self.height = self.pid[0]['shape'][1]
        self.width = self.pid[0]['shape'][2]

        self.input_mean = 127.5
        self.input_std = 127.5

        # Load the TensorFlow Lite model.
        self.si, self.sid, self.sod = load_tflite_model(
            self.speed_prediction_model_path
        )

        self.sc = load('../models/std_scaler.bin')

        # Create an instance of the Lucas-Kanade optical flow algorithm
        self.lk_params = dict(
            winSize=(100, 40),
            maxLevel=5,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10, 0.03
            )
        )

        # self.X = []
        # self.Y = []

        self.lane_speeds = {i: 0 for i in range(1, 4)}

        self.prev_gray = None
        self.prev_pts = None
        self.frame_num = 0

    def browse(self, instance):
        file_path = filechooser.open_file(title="Choose a video file")
        if file_path:
            file_path = file_path[0]  # Assuming single file selection
            try:
                # Set up OpenCV VideoCapture
                self.in_video_path = file_path
                self.init()
                Clock.schedule_interval(self.update, 1.0 / 30.0)  # Update every 1/30th of a second

            except Exception as e:
                print("Error loading video:", repr(e))
        else:
            print("No file selected.")

    def update(self, dt):
        frame = self.predict_and_visualize()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf = frame.tobytes()
            texture = self.image.texture
            if texture is None:
                texture = self.image.texture = Texture.create(
                    size=(frame.shape[1], frame.shape[0]), colorfmt='rgb'
                )
                texture.flip_vertical()
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.image.canvas.ask_update()

    def predict_and_visualize(
        self, verbose=1, min_pixel_speed=15
    ):

        if (self.frame_num < self.max_frames):
            ret, frame = self.cap.read()
            if not ret:
                return None
            self.frame_num += 1
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imH, imW, _ = frame.shape
            image_resized = cv2.resize(image_rgb, (self.width, self.height))
            input_data = np.expand_dims(image_resized, axis=0)

            # Normalize pixel values if using a floating model
            # (i.e. if model is non-quantized)
            if self.pid[0]['dtype'] == np.float32:
                input_data = (np.float32(input_data) - self.input_mean) / self.input_std

            # Perform the detection by running the model with the frame as input
            self.pi.set_tensor(
                self.pid[0]['index'], input_data)
            self.pi.invoke()

            # Convert the frame to grayscale for optflow calculation
            self.frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.frame_num > 2 and self.prev_pts:
                self.prev_pts = np.array(self.prev_pts).astype(np.float32)
                self.prev_pts = np.float32(self.prev_pts)
                self.prev_pts = np.expand_dims(self.prev_pts, axis=1)

                # Calculate optical flow using Lucas-Kanade method
                self.next_pts, _, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, self.frame_gray,
                    self.prev_pts, None, **self.lk_params
                )

                # Draw the tracks
                mask = np.zeros_like(frame)
                for (new, old) in zip(self.next_pts, self.prev_pts):
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)
                    if b < 200:
                        continue

                    pixel_speed = ((a-c)**2 + (b-d)**2)**0.5

                    # Prepare input data (example).
                    input_data = np.array(
                        self.sc.transform([[a, b, pixel_speed]]),
                        dtype=np.float32,
                    )

                    # Set input tensor.
                    self.si.set_tensor(
                        self.sid[0]['index'], input_data)

                    # Run inference.
                    self.si.invoke()

                    # Get output tensor.
                    predicted_speed = self.si.get_tensor(
                        self.sod[0]['index']
                    )[0][0]

                    mask = cv2.line(mask, (a, b), (c, d), (255, 0, 0), 2)
                    frame = cv2.circle(frame, (a, b), 5, (255, 0, 0), -1)
                    cv2.putText(
                        frame,
                        str(predicted_speed)[:4],
                        (a, b),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 255),
                        4,
                    )  # Draw label text

                    cv2.putText(
                        frame,
                        str(predicted_speed)[:4],
                        (get_lane(a, b)*400, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 255),
                        4,
                    )  # Draw label text

                    frame = cv2.add(frame, mask)

                    if self.frame_num in self.vehicles:
                        ((x, y, w, h), real_speed, lane) = self.vehicles[self.frame_num]
                        self.lane_speeds[lane] = real_speed
                        car_id = get_id(x, y)
                        if car_id == get_id(a, b):

                            if (verbose):
                                print('frame_num= ', self.frame_num, end=', ')
                                print('speeds= ', real_speed, pixel_speed)

                            # for _ in range(real_data_coef):
                            #     X.append(
                            #         (frame_num, x, y, pixel_speed, predicted_speed)
                            #     )
                            #     Y.append(real_speed)

                    elif pixel_speed > min_pixel_speed and get_lane(a, b) < 4:
                        # X.append((frame_num, a, b, pixel_speed, predicted_speed))
                        # Y.append(lane_speeds[get_lane(a, b)])

                        if (verbose):
                            print(
                                'frame_num= ', self.frame_num,
                                'lane=', get_lane(a, b),
                                'lane_speeds= ',
                                self.lane_speeds[get_lane(a, b)],
                                'pixel_speed=', pixel_speed
                            )

            output = self.pi.get_tensor(self.pod[0]['index'])
            output = output[0]
            output = output.T

            # Get coordinates of bounding box, first 4 columns of output tensor
            boxes_xywh, scores = output[..., :4], output[..., 4]

            # Threshold Setting
            self.threshold = 0.09

            self.prev_pts = []
            cars = {}
            for box, score in zip(boxes_xywh, scores):
                if score >= self.threshold:
                    x_center, y_center, w, h = box
                    if y_center > 250/imW and w < 100/imH and h < 100/imW:
                        # calculate average box for each car
                        car_id = get_id(x_center*imW, y_center*imH)
                        sum_box, cnt = cars.get(car_id, (((0, 0, 0, 0), 0)))
                        cars[car_id] = (sum_box + box, cnt+1)

            for box, cnt in cars.values():
                x_center, y_center, w, h = box / cnt
                self.prev_pts.append([x_center * imW, y_center * imH])

                x1 = int((x_center - w / 2) * imW)
                y1 = int((y_center - h / 2) * imH)
                x2 = int((x_center + w / 2) * imW)
                y2 = int((y_center + h / 2) * imH)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (10, 255, 0), 2)

            self.prev_gray = self.frame_gray.copy()

            return frame
            self.out.write(frame)

        return None
        # return X, Y

    def on_stop(self):
        # Release OpenCV VideoCapture when the app stops
        self.cap.release()
        self.out.release()


if __name__ == '__main__':
    VideoApp().run()
