"""CVS project: one Picamera2 feed, face recognition and TFLite detection.

Run on the Raspberry Pi: python3 main.py
Required beside this file: encodings.pickle, models/model.tflite,
models/labels.txt. Install dependencies in one Pi virtual environment.

Starting points (cite these in your report):
https://github.com/carolinedunn/facial_recognition
https://github.com/armaanpriyadarshan/TensorFlow-2-Lite-Object-Detection-on-the-Raspberry-Pi
https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi

This initial version displays results and logs CSVs. It does not record video,
generate plots, or implement the assignment's image-quality assessment yet.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import pickle
import time

import cv2
import face_recognition
import numpy as np
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=BASE_DIR / 'models/model.tflite')
    parser.add_argument('--labels', type=Path, default=BASE_DIR / 'models/labels.txt')
    parser.add_argument('--encodings', type=Path, default=BASE_DIR / 'encodings.pickle')
    parser.add_argument('--duration', type=float, default=20.0,
                        help='Observation seconds after camera warm-up')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--face-tolerance', type=float, default=0.6)
    parser.add_argument('--face-scale', type=int, default=4,
                        help='Divide camera dimensions by this for face processing')
    parser.add_argument('--resolution', default='1280x720')
    parser.add_argument('--brightness', type=float, default=0.0)
    parser.add_argument('--contrast', type=float, default=1.0)
    parser.add_argument('--output-dir', type=Path, default=BASE_DIR / 'results')
    # Output positions follow the original SSD lab script. Other models may
    # require a different order; inspect get_output_details() before changing.
    parser.add_argument('--output-order', default='0,1,2',
                        help='Output positions for boxes, classes, scores')
    args = parser.parse_args()
    try:
        args.camera_width, args.camera_height = map(int, args.resolution.split('x'))
        args.output_order = tuple(map(int, args.output_order.split(',')))
    except ValueError:
        parser.error('Use --resolution 1280x720 and --output-order 0,1,2')
    if (min(args.camera_width, args.camera_height) <= 0 or args.face_scale < 1
            or args.duration <= 0 or not 0 <= args.threshold <= 1
            or args.face_tolerance <= 0 or len(args.output_order) != 3
            or min(args.output_order) < 0 or len(set(args.output_order)) != 3):
        parser.error('Invalid resolution, scale, duration, threshold or output order')
    return args


def load_faces(path):
    # Only load an encodings file you created/trust: pickle can execute code.
    with path.open('rb') as handle:
        data = pickle.load(handle)
    encodings = np.asarray(data['encodings'])
    names = list(data['names'])
    if encodings.ndim != 2 or encodings.shape[1] != 128 or len(names) != len(encodings):
        raise ValueError('Face encodings must contain matching names and 128-value vectors')
    return encodings, names


def recognise_faces(rgb, known_encodings, known_names, scale, tolerance):
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (max(1, w // scale), max(1, h // scale)))
    locations = face_recognition.face_locations(small, model='hog')
    encodings = face_recognition.face_encodings(small, locations)
    results = []
    for (top, right, bottom, left), encoding in zip(locations, encodings):
        distances = face_recognition.face_distance(known_encodings, encoding)
        best = int(np.argmin(distances))
        distance = float(distances[best])
        name = known_names[best] if distance <= tolerance else 'Unknown'
        # Lab 4's heuristic: a similarity percentage, NOT a probability.
        score = float(np.clip(1.0 - distance, 0.0, 1.0))
        x_scale, y_scale = w / small.shape[1], h / small.shape[0]
        box = (int(left * x_scale), int(top * y_scale),
               int(right * x_scale), int(bottom * y_scale))
        results.append(dict(kind='face', label=name, score=score,
                            distance=distance, box=box))
    return results


class ObjectDetector:
    def __init__(self, model, labels_path, output_order):
        self.interpreter = tflite.Interpreter(model_path=str(model))
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.outputs = self.interpreter.get_output_details()
        self.order = output_order
        if max(output_order) >= len(self.outputs):
            raise ValueError('Output order does not match this model')
        shape = self.input['shape']
        if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
            raise ValueError('Expected a model input shaped [1, height, width, 3]')
        self.height, self.width = map(int, shape[1:3])
        if self.input['dtype'] not in (np.float32, np.uint8):
            raise ValueError('This lab-based preprocessing supports float32 or uint8 inputs only')
        with labels_path.open() as handle:
            self.labels = [line.strip() for line in handle]
        # Preserve label indices: do not remove placeholder lines such as ???.
        print('Model input:', shape, self.input['dtype'])
        print('Model outputs:', [(x['name'], x['shape'].tolist()) for x in self.outputs])

    def detect(self, rgb, threshold):
        image = cv2.resize(rgb, (self.width, self.height))
        data = np.expand_dims(image, axis=0)
        if self.input['dtype'] == np.float32:
            # Matches the original lab model. Other float models can require
            # different normalization, so check their model documentation.
            data = (data.astype(np.float32) - 127.5) / 127.5
        self.interpreter.set_tensor(self.input['index'], data)
        self.interpreter.invoke()
        boxes, classes, scores = [
            self.interpreter.get_tensor(self.outputs[pos]['index'])[0]
            for pos in self.order
        ]
        if boxes.ndim != 2 or boxes.shape[1] != 4 or classes.ndim != 1 or scores.ndim != 1:
            raise ValueError('Expected SSD outputs: boxes [N,4], classes [N], scores [N]. '
                             'Check --output-order and your model.')
        h, w = rgb.shape[:2]
        results = []
        for box, class_id, score in zip(boxes, classes, scores):
            if not threshold < float(score) <= 1.0:
                continue
            idx = int(class_id)
            if not 0 <= idx < len(self.labels):
                raise ValueError('Class index outside labels file; check model/label pairing')
            ymin, xmin, ymax, xmax = np.clip(box, 0, 1)
            coords = (int(xmin * (w - 1)), int(ymin * (h - 1)),
                      int(xmax * (w - 1)), int(ymax * (h - 1)))
            if coords[2] <= coords[0] or coords[3] <= coords[1]:
                continue
            results.append(dict(kind='object', label=self.labels[idx],
                                score=float(score), distance='', box=coords))
        return results


def draw_results(frame, detections):
    for result in detections:
        left, top, right, bottom = result['box']
        colour = (255, 0, 0) if result['kind'] == 'face' else (0, 255, 0)
        text = f"{result['label']}: {result['score'] * 100:.1f}%"
        cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)
        cv2.putText(frame, text, (left, max(20, top - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)


def main():
    args = parse_args()
    known_encodings, known_names = load_faces(args.encodings)
    detector = ObjectDetector(args.model, args.labels, args.output_order)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    detections_path = args.output_dir / f'{run_id}_detections.csv'
    frames_path = args.output_dir / f'{run_id}_frames.csv'
    camera = Picamera2()
    try:
        # XRGB8888 is stored as B,G,R,X bytes; explicitly convert to BGR/RGB.
        camera.configure(camera.create_preview_configuration(main={
            'format': 'XRGB8888', 'size': (args.camera_width, args.camera_height)}))
        camera.set_controls({'Brightness': args.brightness, 'Contrast': args.contrast})
        camera.start()
        time.sleep(2.0)  # Warm up BEFORE starting the observation timer.
        cv2.namedWindow('CVS Project', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('CVS Project', 1280, 720)
        with detections_path.open('w', newline='') as detection_file, frames_path.open('w', newline='') as frame_file:
            detections_log = csv.writer(detection_file)
            frames_log = csv.writer(frame_file)
            detections_log.writerow(['frame_id', 'time_s', 'kind', 'label', 'score_0_to_1',
                                     'face_distance', 'xmin', 'ymin', 'xmax', 'ymax'])
            frames_log.writerow(['frame_id', 'capture_time_s', 'processing_ms', 'faces', 'objects'])
            start = time.perf_counter()
            frame_id = 0
            print('Observation started. Press q to stop early.')
            while time.perf_counter() - start < args.duration:
                loop_start = time.perf_counter()
                raw = camera.capture_array()
                captured_at = time.perf_counter() - start
                if captured_at >= args.duration:
                    break
                frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Both models see the original image, before any boxes are drawn.
                faces = recognise_faces(rgb, known_encodings, known_names,
                                        args.face_scale, args.face_tolerance)
                objects = detector.detect(rgb, args.threshold)
                for result in faces + objects:
                    detections_log.writerow([frame_id, captured_at, result['kind'],
                                             result['label'], result['score'],
                                             result['distance'], *result['box']])
                processing_ms = (time.perf_counter() - loop_start) * 1000
                # Every processed frame is logged, including frames with NO detections.
                # Missing labels in a frame can therefore be plotted as gaps.
                frames_log.writerow([frame_id, captured_at, processing_ms, len(faces), len(objects)])
                draw_results(frame, faces + objects)
                cv2.putText(frame, f'{captured_at:.1f}s | Processing: {1000 / processing_ms:.1f} FPS',
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow('CVS Project', frame)
                frame_id += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        camera.close()
        cv2.destroyAllWindows()
    print('Saved:', detections_path)
    print('Saved:', frames_path)


if __name__ == '__main__':
    main()


