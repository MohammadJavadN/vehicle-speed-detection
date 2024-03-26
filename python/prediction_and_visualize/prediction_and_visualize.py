import cv2
import csv
import numpy as np
from joblib import load
import xml.etree.ElementTree as ET
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


def load_tflite_model(path):
    interpreter = Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def predict_and_visualize(
    in_video_path, out_video_path, vehicles, plate_model_path,
    speed_model_path, real_data_coef=50, verbose=0,
    min_pixel_speed=15, max_frames=None
):

    cap = cv2.VideoCapture(in_video_path)
    if not max_frames:
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    pi, pid, pod = load_tflite_model(plate_model_path)
    height = pid[0]['shape'][1]
    width = pid[0]['shape'][2]

    input_mean = 127.5
    input_std = 127.5

    # Load the TensorFlow Lite model.
    si, sid, sod = load_tflite_model(speed_model_path)

    sc = load('../models/std_scaler.bin')

    # Create an instance of the Lucas-Kanade optical flow algorithm
    lk_params = dict(
        winSize=(100, 40),
        maxLevel=5,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
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
        if pid[0]['dtype'] == np.float32:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the detection by running the model with the frame as input
        pi.set_tensor(
            pid[0]['index'], input_data)
        pi.invoke()

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

            # Draw the tracks
            mask = np.zeros_like(frame)
            for (new, old) in zip(next_pts, prev_pts):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                if b < 200:
                    continue

                pixel_speed = ((a-c)**2 + (b-d)**2)**0.5

                # Prepare input data (example).
                input_data = np.array(
                    sc.transform([[a, b, pixel_speed]]), dtype=np.float32
                )

                # Set input tensor.
                si.set_tensor(
                    sid[0]['index'], input_data)

                # Run inference.
                si.invoke()

                # Get output tensor.
                predicted_speed = si.get_tensor(
                    sod[0]['index'])[0][0]

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

                if frame_num in vehicles:
                    ((x, y, w, h), real_speed, lane) = vehicles[frame_num]
                    lane_speeds[lane] = real_speed
                    car_id = get_id(x, y)
                    if car_id == get_id(a, b):

                        if (verbose):
                            print('frame_num= ', frame_num, end=', ')
                            print('speeds= ', real_speed, pixel_speed)

                        for _ in range(real_data_coef):
                            X.append(
                                (frame_num, x, y, pixel_speed, predicted_speed)
                            )
                            Y.append(real_speed)

                elif pixel_speed > min_pixel_speed and get_lane(a, b) < 4:
                    X.append((frame_num, a, b, pixel_speed, predicted_speed))
                    Y.append(lane_speeds[get_lane(a, b)])

                    if (verbose):
                        print(
                            'frame_num= ', frame_num,
                            'lane=', get_lane(a, b),
                            'lane_speeds= ', lane_speeds[get_lane(a, b)],
                            'pixel_speed=', pixel_speed
                        )

        output = pi.get_tensor(pod[0]['index'])
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
                if y_center > 250/imW and w < 100/imH and h < 100/imW:
                    # calculate average box for each car
                    car_id = get_id(x_center*imW, y_center*imH)
                    sum_box, cnt = cars.get(car_id, (((0, 0, 0, 0), 0)))
                    cars[car_id] = (sum_box + box, cnt+1)

        for box, cnt in cars.values():
            x_center, y_center, w, h = box / cnt
            prev_pts.append([x_center * imW, y_center * imH])

            x1 = int((x_center - w / 2) * imW)
            y1 = int((y_center - h / 2) * imH)
            x2 = int((x_center + w / 2) * imW)
            y2 = int((y_center + h / 2) * imH)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (10, 255, 0), 2)

        prev_gray = frame_gray.copy()

        frame2 = cv2.resize(frame, (690, 540))
        cv2.imshow('output', frame2)
        out.write(frame)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()

    return X, Y


def save_data_in_file(X, Y, path='../data/data2.csv'):
    # Define field names (column headers)
    field_names = [
        'frame_nmr', 'x', 'y', 'pixel_speed',
        'real_speed', 'prediction_speed'
    ]

    # Create a list of dictionaries (rows)
    rows = []
    for i, (fn, x, y, ps, prediction) in enumerate(X):
        rows.append(
            {
                'frame_nmr': fn,
                'x': x,
                'y': y,
                'pixel_speed': ps,
                'prediction_speed': prediction,
                'real_speed': Y[i],
            }
        )

    # Write data to CSV file
    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()  # Write column headers
        writer.writerows(rows)  # Write data rows

    print(f"CSV file {path} created successfully!")


# Main function
def main():
    xml_path = '../data/vehicles.xml'
    in_video_path = '../data/video.h264'
    out_video_path = '../data/out.mp4'
    license_plate_detector_model_path = '' +\
        '../models/license_plate_detector_float32.tflite'
    speed_prediction_model_path = '../models/speed_prediction_model.tflite'

    vehicles = parse_xml(xml_path)
    print('xml file was parsed successfully!\n')

    X, y = predict_and_visualize(
        in_video_path=in_video_path,
        out_video_path=out_video_path,
        vehicles=vehicles,
        speed_model_path=speed_prediction_model_path,
        plate_model_path=license_plate_detector_model_path,
        max_frames=5000,
        verbose=1,
    )
    print('\n prediction is completed!')

    save_data_in_file(X, y)


if __name__ == "__main__":
    main()
