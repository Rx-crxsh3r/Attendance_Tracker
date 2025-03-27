import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

#load YOLO(we chose v8) model
model_path = r'C:\Users\thebl\Desktop\‎\folder1\Face recoginition(robo,YOLO)\best.pt' #Change this if needed
model = YOLO(model_path)

st.title("Object Detection with YOLO(classroom attendance tracker)")

#sidebar
option = st.radio("Choose an option", ["Upload Image", "Live camera"])

#detecting people as an image
def detect_people(frame):
    results = model(frame) #YOLO detection awakened :shaking_face:
    detected_objects = results[0].boxes
    num_students = 0

    for box in detected_objects:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        if cls == 0:
            num_students += 1
            label = f"Person: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    return frame, num_students

#upload image option
if option == "Upload Image":
    st.subheader("Upload an image for attendance detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])



    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        #convert image to numpy array and process
        image_np = np.array(image)
        processed_image, count = detect_people(image_np)


        #show results
        st.image(processed_image, caption=f"Detected attendance: {count}", use_container_width=True)
        st.write(f"### Attendance Count: {count}")


################################ or could be:
    # image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    # if image_file is not None:
    #     image = Image.open(image_file)
    #     frame = np.array(image)
    #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     result_frame, num_students = detect_people(frame)
    #     st.image(result_frame, channels="BGR")
    #     st.write(f"Number of students detected: {num_students}")

#live camera option
elif option == "Live Camera":
    st.subheader("Live camera feed for attendance detection")
    st.write("Click 'Start Camera' to begin tracking")

    #camera control buttons
    start = st.button("Start Camera")
    stop = st.button("Stop Camera")

    if start:
        cap = cv2.VideoCapture(0) #open webcam
        frame_placeholder = st.empty() #placeholder for updating frames
        attendance_text = st.empty() #placeholder for updating attendance count

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read frame")
                break

            #process frame
            processed_frame, num_students = detect_people(frame)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB) #Convert to RGB for streamlit
        
            #update UI
            frame_placeholder.image(processed_frame, caption="Live Attendance tracking", use_container_width=True)
            attendance_text.write(f" ### Current Attendance Count: {num_students}")

            if stop:
                break

        cap.release()
        cv2.destroyAllWindows()



