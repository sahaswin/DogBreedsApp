import cv2
from ultralytics import YOLO
import argparse

# Argument parser to choose image or webcam
parser = argparse.ArgumentParser(description='YOLO object detection')
parser.add_argument('--source', type=str, help='Path to image file or "webcam" for webcam detection', default='webcam')
args = parser.parse_args()

# Load YOLO model
model = YOLO('best.pt')


def detect_image(image_path):
    image = cv2.imread(image_path)
    results = model.predict(source=image)
    draw_boxes(image, results)
    cv2.imshow('YOLO Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame)
        draw_boxes(frame, results)
        cv2.imshow('YOLO Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def draw_boxes(frame, results):
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Convert tensor to numpy array
        confidences = result.boxes.conf.cpu().numpy()  # Convert tensor to numpy array
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Convert tensor to numpy array

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[class_id]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Display label and confidence
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)


if __name__ == "__main__":
    if args.source == 'webcam':
        detect_webcam()
    else:
        detect_image(args.source)


