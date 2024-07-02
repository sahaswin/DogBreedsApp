import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

# Load YOLO model
model = YOLO('best.pt')


def detect_image(image):
    results = model.predict(image, verbose=False)
    draw_boxes(image, results)
    return image


def draw_boxes(frame, results):
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Convert tensor to numpy array
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[class_id]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            # Display label and confidence
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0),
                        2)


st.title("YOLO Object Detection")
st.write("Choose to upload an image or use your webcam for object detection.")

option = st.selectbox("Select input method", ("Upload Image", "Use Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Read the image
        image = cv2.imread(tmp_file_path)
        # Perform detection
        detected_image = detect_image(image)
        detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
        # Display the image
        st.image(detected_image_rgb, caption="Detected Image", use_column_width=True)

if option == "Use Webcam":
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    stop_button_key = "stop_button0"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        detected_frame = detect_image(frame)
        detected_frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
        # Display the frame
        stframe.image(detected_frame_rgb, channels="RGB")

    cap.release()
    cv2.destroyAllWindows()
