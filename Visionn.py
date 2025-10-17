import streamlit as st
import cv2
import seaborn as sns
import numpy as np
from PIL import Image
from datetime import datetime
import time
import os

sns.set(style='darkgrid')
st.set_page_config(page_title="Smart Vision Detection Console", layout="wide")

current_hour = datetime.now().hour
greeting = "Good Morning" if current_hour < 12 else "Good Afternoon" if 12 <= current_hour < 18 else "Good Evening"
st.markdown(f"<h1 style='text-align: center; color: #ff7f50;'>{greeting}</h1>", unsafe_allow_html=True)

progress_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.010)
    progress_bar.progress(percent_complete + 1)

cfg_file = 'yolov3.cfg'
weights_file = 'yolov3.weights'

if not os.path.exists(cfg_file):
    st.error(f"Error: File {cfg_file} not found.")
    st.stop()
if not os.path.exists(weights_file):
    st.error(f"Error: File {weights_file} not found.")
    st.stop()

try:
    net = cv2.dnn.readNet(weights_file, cfg_file)
    st.success("Model Configuration Successfully Loaded.")
except cv2.error as e:
    st.error(f"Error Loading Model: {e}")
    st.stop()

try:
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    st.error("coco.names file not found.")
    st.stop()

layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers()
if len(unconnected_out_layers.shape) == 2:
    output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
else:
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

st.sidebar.title("Settings")
st.sidebar.write("Configure detection settings below:")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
nms_threshold = st.sidebar.slider("NMS Threshold", 0.1, 1.0, 0.4)
enable_alert = st.sidebar.checkbox("Enable Alerts for Specific Classes", value=True)
alert_classes = st.sidebar.multiselect("Alert Classes", classes, default=[])

st.sidebar.write("---")
st.sidebar.write("**Upload an Image or Use Webcam**")
source_option = st.sidebar.radio("Select Source", ("Webcam", "Upload Image"))

if source_option == "Webcam":
    st.write("### Real-time Webcam Object Detection")
    start_webcam = st.button("Start Webcam", key="webcam_start")

    if start_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
        else:
            FRAME_WINDOW = st.image([])

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Error: Could not read frame.")
                    break

                frame = cv2.resize(frame, (640, 480))
                height, width, channels = frame.shape

                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                class_ids = []
                confidences = []
                boxes = []
                detected_alerts = []

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > confidence_threshold:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        confidence = confidences[i]
                        color = colors[class_ids[i]]

                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        if enable_alert and label in alert_classes:
                            detected_alerts.append(label)

                if detected_alerts:
                    st.warning(f"Detected Alerts: {', '.join(detected_alerts)}")

                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            cap.release()

if source_option == "Upload Image":
    st.write("### Image Object Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        img = cv2.resize(img, (640, 480))
        height, width, channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)

