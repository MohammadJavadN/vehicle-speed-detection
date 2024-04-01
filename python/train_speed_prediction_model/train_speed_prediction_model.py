import cv2
import csv
import os
import numpy as np
from joblib import dump
import tensorflow as tf
import xml.etree.ElementTree as ET
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.lite.python.interpreter import Interpreter


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


def extract_augmented_data_top_view_with_plate(
        video_path, vehicles, modelpath, real_data_coef=50,
        verbose=0, min_pixel_speed=15, max_frames=None):

    cap = cv2.VideoCapture(video_path)
    if not max_frames:
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Create an instance of the Lucas-Kanade optical flow algorithm
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=0,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 0.001)
    )

    X = []
    Y = []

    lane_speeds = {i: 0 for i in range(1, 4)}

    prev_gray = None
    prev_pts = None
    frame_num = 0
    while (frame_num < max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imH, imW, _ = frame.shape
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model
        # (i.e. if model is non-quantized)
        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the detection by running the model with the frame as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Convert the frame to grayscale for optflow calculation
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_num > 2 and prev_pts:
            prev_pts = np.array(prev_pts).astype(np.float32)
            prev_pts = np.float32(prev_pts)
            prev_pts = np.expand_dims(prev_pts, axis=1)

            # Calculate optical flow using Lucas-Kanade method
            next_pts, _, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, frame_gray,
                prev_pts, None, **lk_params
            )

            for (new, old) in zip(next_pts, prev_pts):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                pixel_speed = ((a-c)**2 + (b-d)**2)**0.5

                if frame_num in vehicles:
                    ((x, y, w, h), real_speed, lane) = vehicles[frame_num]
                    lane_speeds[lane] = real_speed
                    car_id = get_id(x, y)
                    if car_id == get_id(a, b):

                        if (verbose):
                            print('frame_num= ', frame_num, end=', ')
                            print('speeds= ', real_speed, pixel_speed)

                        for _ in range(real_data_coef):
                            X.append((x, y, pixel_speed))
                            Y.append(real_speed)

                elif (pixel_speed > min_pixel_speed or b < 300)\
                        and get_lane(a, b) < 4:
                    X.append((a, b, pixel_speed))
                    Y.append(lane_speeds[get_lane(a, b)])

                    if (verbose):
                        print(
                            'frame_num= ', frame_num,
                            'lane=', get_lane(a, b),
                            'lane_speeds= ', lane_speeds[get_lane(a, b)],
                            'pixel_speed=', pixel_speed
                        )

        output = interpreter.get_tensor(output_details[0]['index'])
        output = output[0]
        output = output.T

        # Get coordinates of bounding box, first 4 columns of output tensor
        boxes_xywh, scores = output[..., :4], output[..., 4]

        # Threshold Setting
        threshold = 0.09

        prev_pts = []
        cars = {}
        for box, score in zip(boxes_xywh, scores):
            if score >= threshold:
                x_center, y_center, w, h = box
                if w < 100/imH and h < 100/imW:
                    # calculate average box for each car
                    car_id = get_id(x_center*imW, y_center*imH)
                    sum_box, cnt = cars.get(car_id, (((0, 0, 0, 0), 0)))
                    cars[car_id] = (sum_box + box, cnt+1)

        for box, cnt in cars.values():
            x_center, y_center, w, h = box / cnt
            prev_pts.append([x_center * imW, y_center * imH])

        prev_gray = frame_gray.copy()

    cap.release()

    # Agumenting with 0 speeds
    for _ in range(200):
        X.append(
            (
                np.random.randint(imW),
                np.random.randint(imH),
                0,
            )
        )
        Y.append(0)

    return X, Y


def extract_augmented_data_top_view_no_plate(
        video_path, vehicles, real_data_coef=50,
        verbose=0, min_pixel_speed=15, max_frames=None):

    cap = cv2.VideoCapture(video_path)
    if not max_frames:
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create an instance of the Lucas-Kanade optical flow algorithm
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=0,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 0.001)
    )

    X = []
    Y = []

    lane_speeds = {i: 0 for i in range(1, 4)}
    last_update = {i: -10 for i in range(1, 4)}
    prev_gray = None
    prev_pts = None
    frame_num = 0
    while (frame_num < max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        imH, imW, _ = frame.shape

        # Convert the frame to grayscale for optflow calculation
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_num > 2:
            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray, 200, 0.3, 5, blockSize=7,
                useHarrisDetector=False, k=0.04)

            # Calculate optical flow using Lucas-Kanade method
            next_pts, _, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, frame_gray,
                prev_pts, None, **lk_params
            )

            pixel_lane_speeds = [0, 0, 0, 0]
            cnt_of_lane_speeds = [0, 0, 0, 0]

            for (new, old) in zip(next_pts, prev_pts):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                if (b < d or abs(a-c) > imW/100 or abs(b-d) > imH*0.14):
                    continue
                pixel_speed = ((a-c)**2 + (b-d)**2)**0.5
                if pixel_speed < min_pixel_speed or b < 0.25 * imH:
                    continue
                lane = get_lane(a, b)
                pixel_lane_speeds[lane-1] += pixel_speed
                cnt_of_lane_speeds[lane-1] += 1

            if frame_num in vehicles:
                ((x, y, w, h), real_speed, lane) = vehicles[frame_num]
                lane_speeds[lane] = real_speed
                last_update[lane] = frame_num

            for lane in range(1, 4):
                if frame_num - last_update[lane] > 40:
                    continue

                if cnt_of_lane_speeds[lane-1] > 0:
                    pixel_speed = pixel_lane_speeds[lane-1]\
                        / cnt_of_lane_speeds[lane-1]
                else:
                    continue

                real_speed = lane_speeds[lane]

                if (verbose):
                    print('frame_num= ', frame_num, end=', ')
                    print('speeds= ', lane_speeds[lane], pixel_speed)

                if frame_num in vehicles:
                    for _ in range(real_data_coef):
                        X.append((a, b, pixel_speed))
                        Y.append(real_speed)
                else:
                    X.append((a, b, pixel_speed))
                    Y.append(real_speed)

        prev_gray = frame_gray.copy()

    cap.release()

    # Agumenting with 0 speeds
    for _ in range(50):
        X.append(
            (
                np.random.randint(imW),
                np.random.randint(imH),
                0,
            )
        )
        Y.append(0)

    return X, Y


def extract_augmented_data_side_view_no_plate(
        video_dir, real_speed, verbose=0,
        min_pixel_speed=15):

    # Create an instance of the Lucas-Kanade optical flow algorithm
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=0,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 0.001)
    )

    X = []
    Y = []

    prev_gray = None
    prev_pts = None

    for file in os.listdir(video_dir):
        if verbose:
            print('file:', file)
        if file.endswith(".mp4"):
            video_path = os.path.join(video_dir, file)
            if verbose:
                print('video_path:', video_path)
            real_speed = int(video_path.split('_')[-1].split('.')[0])
            if verbose:
                print('real_speed:', real_speed)
            cap = cv2.VideoCapture(video_path)

            frame_num = 0
            while (True):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_num += 1
                imH, imW, _ = frame.shape

                # Convert the frame to grayscale for optflow calculation
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if frame_num > 2:
                    prev_pts = cv2.goodFeaturesToTrack(
                        prev_gray, 200, 0.3, 5, blockSize=7,
                        useHarrisDetector=False, k=0.04)

                    # Calculate optical flow using Lucas-Kanade method
                    next_pts, _, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray, frame_gray,
                        prev_pts, None, **lk_params
                    )

                    sum_pixel_speed = 0
                    cnt = 0
                    for (new, old) in zip(next_pts, prev_pts):
                        a, b = new.ravel().astype(int)
                        c, d = old.ravel().astype(int)
                        pixel_speed = ((a-c)**2 + (b-d)**2)**0.5
                        if pixel_speed < min_pixel_speed:
                            continue
                        sum_pixel_speed += pixel_speed
                        cnt += 1

                    if cnt > 0:
                        if (verbose):
                            print('frame_num= ', frame_num, end=', ')
                            print('speeds= ', real_speed, sum_pixel_speed/cnt)

                        X.append((a, b, sum_pixel_speed/cnt))
                        Y.append(real_speed)
                prev_gray = frame_gray.copy()

            cap.release()

    # Agumenting with 0 speeds
    for _ in range(50):
        X.append(
            (
                np.random.randint(imW),
                np.random.randint(imH),
                0,
            )
        )
        Y.append(0)

    return X, Y


def save_data_in_file(X, Y, path='../../data/data.csv'):
    # Define field names (column headers)
    field_names = ['x', 'y', 'pixel_speed', 'real_speed']

    # Create a list of dictionaries (rows)
    rows = []
    for i, (x, y, ps) in enumerate(X):
        rows.append(
            {
                'x': x,
                'y': y,
                'pixel_speed': ps,
                'real_speed': Y[i],
            }
        )

    # Write data to CSV file
    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()  # Write column headers
        writer.writerows(rows)  # Write data rows

    print(f"CSV file {path} created successfully!")


def train(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    print(len(X_train), len(X_test))
    # Normalize the input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the neural network model
    model = Sequential([
        Dense(1024, input_shape=(3,), activation='relu'),
        Dense(1024, activation='relu'),
        Dense(1)  # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(
        X_train_scaled, np.array(y_train),
        epochs=100, batch_size=32, verbose=0
    )

    # Make predictions on the test set
    predictions = model.predict(X_test_scaled).flatten()

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error:", mse)

    precision = 0
    for i, p in enumerate(predictions):
        precision += (1 - (abs(y_test[i] - p)/p))

    print('precision=', precision/len(predictions))

    return model, scaler


def save_model(model, model_path):
    model.export("../model/pb_model", "tf_saved_model")
    converter = tf.lite.TFLiteConverter.from_saved_model("../model/pb_model")
    tflite_model = converter.convert()
    with open(model_path, "wb") as f:
        f.write(tflite_model)

    print(f"The model {model_path} saved successfully!")


# Main function
def main_top_with_plate():
    xml_path = '../../data/vehicles.xml'
    video_path = '../../data/video.h264'
    license_plate_detector_model_path = '' +\
        '../../models/license_plate_detector_float32.tflite'
    speed_prediction_model_path = '../../models/' + \
        'speed_prediction_top_view_model.tflite'

    vehicles = parse_xml(xml_path)
    print('xml file was parsed successfully!\n')

    X, y = extract_augmented_data_top_view_with_plate(
        video_path=video_path,
        vehicles=vehicles,
        modelpath=license_plate_detector_model_path,
        max_frames=5000,
        verbose=1,
    )
    print('\naugmented data extracted successfully!')

    save_data_in_file(X, y, path="../../data/top_view_data.csv")

    print('\ntraining model started...')

    model, scaler = train(X, y)

    dump(scaler, '../../models/std_scaler.bin', compress=True)

    print('\nModel trained!')

    save_model(model, speed_prediction_model_path)


# Main function
def main_top_no_plate():
    xml_path = '../../data/vehicles.xml'
    video_path = '../../data/video.h264'
    speed_prediction_model_path = '../../models/' + \
        'speed_prediction_top_view_no_plate_model.tflite'

    vehicles = parse_xml(xml_path)
    print('xml file was parsed successfully!\n')

    X, y = extract_augmented_data_top_view_no_plate(
        video_path=video_path,
        vehicles=vehicles,
        max_frames=5000,
        verbose=1,
    )
    print('\naugmented data extracted successfully!')

    save_data_in_file(X, y, path="../../data/top_view_no_plate_data.csv")

    print('\ntraining model started...')

    model, scaler = train(X, y)

    dump(
        scaler, '../../models/std_scaler_top_view_no_plate.bin',
        compress=True,
    )

    print('\nModel trained!')

    save_model(model, speed_prediction_model_path)


# Main function
def main_side_no_plate():
    video_dir = '../../data/sideView'
    speed_prediction_model_path = '../../models/' +\
        'speed_prediction_side_view_no_plate_model.tflite'

    X, y = extract_augmented_data_side_view_no_plate(
        video_dir=video_dir,
        real_speed=80,
        verbose=1,
    )
    print('\naugmented data extracted successfully!')

    save_data_in_file(X, y, path="../../data/side_view_no_plate_data.csv")

    print('\ntraining model started...')

    model, scaler = train(X, y)

    dump(scaler, '../../models/std_scaler_side_view.bin', compress=True)

    print('\nModel trained!')

    save_model(model, speed_prediction_model_path)


if __name__ == "__main__":
    # main()
    # main_side_no_plate()
    main_top_no_plate()
