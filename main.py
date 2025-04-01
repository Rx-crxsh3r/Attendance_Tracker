import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import datetime
import base64
import time

# Load YOLO (we chose v8) model
model_path = r'C:\Users\thebl\Desktop\‎\folder1\Face recoginition(robo,YOLO)\best.pt'  # Change this if needed
model = YOLO(model_path)

# Set the title of the Streamlit app
st.title("Object Detection with YOLO (Classroom Attendance Tracker)")

# Create a sidebar with options
option = st.sidebar.radio("Choose an option", ["Upload Image", "Live Camera", "Dashboard"])

# Initialize attendance data
attendance_data = pd.DataFrame(columns=["Date", "Time", "Attendance Count"])

# Function to detect people in an image or frame
def detect_people(frame):
    """
    Detect people in the given frame using the YOLO model.

    Args:
    frame (numpy array): The input frame or image.

    Returns:
    processed_frame (numpy array): The frame with detected people marked.
    num_students (int): The number of students detected.
    """
    results = model(frame)  # Perform YOLO detection
    detected_objects = results[0].boxes
    num_students = 0

    for box in detected_objects:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        if cls == 0:  # Assuming class 0 is for people
            num_students += 1
            label = f"Person: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, num_students

# Option to upload an image
if option == "Upload Image":
    st.subheader("Upload an image for attendance detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Convert image to numpy array and process
        image_np = np.array(image)
        processed_image, count = detect_people(image_np)

        # Record attendance data
        current_time = datetime.datetime.now()
        attendance_data.loc[len(attendance_data)] = [current_time.date(), current_time.time(), count]

        # Show results
        st.image(processed_image, caption=f"Detected attendance: {count}", use_container_width=True)
        st.write(f"### Attendance Count: {count}")

# Option to use the live camera with countdown feature
elif option == "Live Camera":
    st.subheader("Live camera feed for attendance detection")
    st.write("Click 'Start Camera' to begin tracking")

    # Camera control buttons
    start = st.button("Start Camera")
    stop = st.button("Stop Camera")

    countdown = 0  # Countdown timer (5 seconds)
    current_count = 0  # Current number of detected students

    if start:
        cap = cv2.VideoCapture(0)  # Open webcam
        frame_placeholder = st.empty()  # Placeholder for updating frames
        attendance_text = st.empty()  # Placeholder for updating attendance count
        countdown_text = st.empty()  # Placeholder for countdown display

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read frame")
                break

            # Process frame
            processed_frame, num_students = detect_people(frame)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Streamlit

            # Update UI
            frame_placeholder.image(processed_frame, caption="Live Attendance Tracking", use_container_width=True)
            attendance_text.write(f"### Current Attendance Count: {num_students}")

            # Countdown logic
            if num_students > 0:
                if countdown < 5:
                    countdown += 1
                    countdown_text.write(f"### Countdown: {5 - countdown + 1} seconds")
                    time.sleep(1)  # Wait for 1 second
                else:
                    # Save attendance data
                    current_time = datetime.datetime.now()
                    attendance_data.loc[len(attendance_data)] = [current_time.date(), current_time.time(), num_students]
                    st.success(f"Attendance saved: {num_students} students detected")
                    countdown = 0  # Reset countdown
            else:
                countdown = 0  # Reset countdown if no students are detected
                countdown_text.write("### Countdown: Reset")

            if stop:
                break

        cap.release()
        cv2.destroyAllWindows()

# Dashboard option
elif option == "Dashboard":
    st.subheader("Attendance Dashboard")

    # Check if attendance data is available
    if attendance_data.empty:
        st.warning("No attendance data available. Please perform attendance detection first.")
    else:
        # Show attendance trends over time
        st.write("### Attendance Trends Over Time")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=attendance_data.index, y="Attendance Count", data=attendance_data, ax=ax)
        ax.set_xlabel("Time")
        ax.set_ylabel("Attendance Count")
        ax.set_title("Attendance Trends")
        st.pyplot(fig)

        # Show bar graph of attendance
        st.write("### Daily Attendance Summary")
        daily_summary = attendance_data.groupby("Date")["Attendance Count"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Date", y="Attendance Count", data=daily_summary, ax=ax)
        ax.set_title("Daily Attendance Summary")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

        # Show heatmap of attendance
        st.write("### Attendance Heatmap")
        # Convert time to hour for heatmap
        attendance_data['Hour'] = pd.to_datetime(attendance_data['Time']).dt.hour
        heatmap_data = attendance_data.pivot_table(index='Hour', columns='Date', values='Attendance Count', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", ax=ax)
        ax.set_title("Attendance Heatmap")
        ax.set_xlabel("Date")
        ax.set_ylabel("Hour of Day")
        st.pyplot(fig)

        # Export attendance data as CSV
        st.write("### Export Attendance Data")
        csv = attendance_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="attendance_data.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)