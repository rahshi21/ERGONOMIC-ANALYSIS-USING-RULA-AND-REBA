import cv2
import mediapipe as mp
import mimetypes
import streamlit as st
import time
from angle_calc import angle_calc
import matplotlib.pyplot as plt
from fpdf import FPDF

# Initialize mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Initialize session state variables if they do not exist
if 'rula_scores' not in st.session_state:
    st.session_state.rula_scores = []

if 'reba_scores' not in st.session_state:
    st.session_state.reba_scores = []

if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

if 'video_processing_done' not in st.session_state:
    st.session_state.video_processing_done = False
if 'video_processing_active' not in st.session_state:
    st.session_state.video_processing_active = False

def image_pose_estimation(name):
    img = cv2.imread(name)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    pose1 = []
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            x_y_z = []
            h, w, c = img.shape
            x_y_z.append(lm.x)
            x_y_z.append(lm.y)
            x_y_z.append(lm.z)
            x_y_z.append(lm.visibility)
            pose1.append(x_y_z)
            cx, cy = int(lm.x*w), int(lm.y*h)
            if id % 2 == 0:
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            else:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
    img = cv2.resize(img, (700, 700))
    st.image(img, caption="Image with Pose Estimation", use_column_width=True)
    rula, reba = angle_calc(pose1)
    st.write(f"Rapid Upper Limb Assessment Score : {rula}")
    st.write(f"Rapid Entire Body Score : {reba}")

def video_pose_estimation(name, process_interval=20):
    cap = cv2.VideoCapture(name)
    frame_count = 0
    global rula_scores, reba_scores  # Use global variables
    rula_scores = []
    reba_scores = []
    frame_times = []

    st.session_state.video_processing_active = True

    while cap.isOpened() and st.session_state.video_processing_active:
        success, img = cap.read()
        if not success:
            break

        frame_count += 1

        # Process one frame every `process_interval` frames
        if frame_count % process_interval == 0:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            pose1 = []
            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    x_y_z = []
                    h, w, c = img.shape
                    x_y_z.append(lm.x)
                    x_y_z.append(lm.y)
                    x_y_z.append(lm.z)
                    x_y_z.append(lm.visibility)
                    pose1.append(x_y_z)
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    if id % 2 == 0:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    else:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                st.image(img, caption=f"Video Frame {frame_count}", use_column_width=True)
                rula, reba = angle_calc(pose1)
                if rula != "NULL" and reba != "NULL":
                    st.session_state.rula_scores.append(int(rula))
                    st.session_state.reba_scores.append(int(reba))
                    st.session_state.frame_count += 1

                    # Debugging: Print scores to ensure they are being captured
                    st.write(f"RULA Score: {rula}, REBA Score: {reba}")
                st.write(f"Processed Frame: {frame_count}, RULA: {rula}, REBA: {reba}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.1)
    
    cap.release()
    st.session_state.video_processing_done = True
    st.write("Video processing stopped. Click the button below to generate a report.")

def generate_report():
    st.write("Generate report function is being called.")
    st.write(f"RULA Scores: {st.session_state.rula_scores}")
    st.write(f"REBA Scores: {st.session_state.reba_scores}")
    rula_scores = st.session_state.rula_scores
    reba_scores = st.session_state.reba_scores
    frame_count = st.session_state.frame_count
    if not rula_scores or not reba_scores:
        st.write("No scores available to generate the report.")
        return

    avg_rula = sum(rula_scores) / len(rula_scores)
    avg_reba = sum(reba_scores) / len(reba_scores)
    
    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add title
    pdf.cell(200, 10, txt="Ergonomic Analysis Report", ln=True, align='C')

    # Add table headers
    pdf.cell(60, 10, txt="Frame Count", border=1)
    pdf.cell(60, 10, txt="RULA Score", border=1)
    pdf.cell(60, 10, txt="REBA Score", border=1)
    pdf.ln()

    # Add table rows
    for i in range(len(rula_scores)):
        pdf.cell(60, 10, txt=str(i + 1), border=1)
        pdf.cell(60, 10, txt=str(rula_scores[i]), border=1)
        pdf.cell(60, 10, txt=str(reba_scores[i]), border=1)
        pdf.ln()
    
    # Add averages
    pdf.cell(60, 10, txt="Averages", border=1)
    pdf.cell(60, 10, txt=f"{avg_rula:.2f}", border=1)
    pdf.cell(60, 10, txt=f"{avg_reba:.2f}", border=1)
    pdf.ln()
    
    # Save PDF
    pdf_output = "ergonomic_analysis_report.pdf"
    pdf.output(pdf_output)
    
    # Provide download link
    with open(pdf_output, "rb") as file:
        st.download_button(
            label="Download Report",
            data=file,
            file_name=pdf_output,
            mime="application/octet-stream"
        )

def home_page():
    st.title("Ergonomic Analysis")
    
    # Welcome Banner
    # st.image("https://example.com/welcome-banner.jpg", use_column_width=True)
    
    st.markdown("""
    Welcome to the Ergonomic Analysis Tool! This application is designed to help you assess and improve workplace ergonomics by analyzing posture using RULA (Rapid Upper Limb Assessment) and REBA (Rapid Entire Body Assessment).
    
    **What You Can Do Here:**
    - **Upload Files:** Upload images or videos for posture analysis.
    - **Start Analysis:** Initiate the analysis to get RULA and REBA scores for the uploaded files.
    - **Generate Reports:** Create detailed reports based on the analysis.

    **Instructions:**
    1. Use the sidebar to upload an image or video file.
    2. Click on "START POSTURE ANALYSIS" to begin the analysis.
    3. Once the analysis is complete, click "GENERATE REPORT" to download the results.

    **How It Helps:**
    This tool helps in evaluating and improving workplace ergonomics, preventing musculoskeletal disorders, and enhancing overall safety and comfort.

    **Features:**
    - Real-time posture analysis
    - Detailed RULA and REBA scoring
    - Downloadable PDF reports

    **Get Started:**
    Use the sidebar to upload your files and start exploring the ergonomic analysis features!
    """)
    
    st.subheader("Upload and Analyze")
    st.markdown("""
    Use the file uploader on the sidebar to select an image or video for analysis. 
    Ensure the file format is correct and follow the on-screen instructions to start the posture analysis.
    """)
    
    # Add a section for FAQs or Additional Help
    st.subheader("Frequently Asked Questions")
    st.markdown("""
    **Q: What file formats are supported?**
    A: You can upload images (JPG, JPEG, PNG) and videos (MP4, AVI) for analysis.

    **Q: How do I start the analysis?**
    A: After uploading your file, click the "START POSTURE ANALYSIS" button to begin.

    **Q: How can I view the results?**
    A: Once the analysis is complete, you can generate and download the report from the sidebar.
    """)
    
    st.subheader("Contact Support")
    st.markdown("""
    If you have any questions or need assistance, feel free to reach out to our support team.
    
    **Email:** support@example.com
    **Phone:** +123-456-7890
    """)

    # Optionally, add an interactive demo or video tutorial
    st.subheader("Watch a Demo")
    st.video("https://example.com/demo-video.mp4", format="mp4")



def details_page1():    
    st.title("RULA (Rapid Upper Limb Assessment)")
    st.markdown("""
    **Significance:**
    The Rapid Upper Limb Assessment (RULA) is a survey method that evaluates the postural risks associated with upper limb disorders. It helps in identifying musculoskeletal disorders in the upper limbs and neck.
    """)
    st.image(r"https://github.com/rahshi21/ERGONOMIC-ANALYSIS-USING-RULA-AND-REBA/blob/main/RULA.png?raw=true", caption="REBA Evaluation Flowchart", use_column_width=True)

    st.image(r"https://github.com/rahshi21/ERGONOMIC-ANALYSIS-USING-RULA-AND-REBA/blob/main/RULA%20SCORE.png?raw=true", caption="RULA Evaluation Score", use_column_width=True)
    

def details_page2():    
    st.title("REBA (Rapid Entire Body Assessment)")
    st.markdown("""
    **Significance:**
    The Rapid Entire Body Assessment (REBA) is used to assess the postural risk associated with the entire body. It is particularly useful in evaluating the risk of musculoskeletal disorders across various body segments.
    """)
    st.image(r"https://github.com/rahshi21/ERGONOMIC-ANALYSIS-USING-RULA-AND-REBA/blob/main/REBA.png?raw=true", caption="RULA Evaluation Flowchart")
    
    st.image(r"https://github.com/rahshi21/ERGONOMIC-ANALYSIS-USING-RULA-AND-REBA/blob/main/REBA%20SCORE.png?raw=true", caption="RULA Evaluation Score", use_column_width=True)


def purpose_page():
    st.title("Purpose of this Tool")
    
    st.markdown("""
    **Introduction:**
    The ergonomic analysis tool is designed to evaluate and improve workplace ergonomics by analyzing postural risks using RULA (Rapid Upper Limb Assessment) and REBA (Rapid Entire Body Assessment).

    **Objectives:**
    - **Assess Postural Risks:** Identify and assess postural risks associated with both upper limbs and the entire body to prevent musculoskeletal disorders.
    - **Enhance Workplace Safety:** Provide actionable insights to improve ergonomic practices and enhance overall workplace safety.
    - **Support Ergonomic Research:** Aid in ergonomic research by providing a robust method for analyzing and visualizing postural data.

    **Benefits:**
    - **Prevent Injuries:** Help prevent repetitive strain injuries and other musculoskeletal disorders by assessing and addressing poor ergonomic practices.
    - **Data-Driven Decisions:** Enable data-driven decisions for ergonomic improvements by providing quantitative analysis of posture.
    - **Improved Comfort:** Enhance employee comfort and productivity by identifying and mitigating ergonomic risks.

    **How It Works:**
    - **Pose Estimation:** Utilizes advanced pose estimation technology to analyze body posture in real-time.
    - **Score Calculation:** Calculates RULA and REBA scores to evaluate postural risks based on the captured pose data.
    - **Generate Reports:** Provides detailed reports and visualizations to help in understanding and improving ergonomic practices.

    **Target Audience:**
    - **Occupational Health Professionals:** Designed for use by occupational health professionals to assess and improve workplace ergonomics.
    - **Ergonomists and Safety Officers:** Useful for ergonomists and safety officers in designing better ergonomic interventions.
    - **Employers and Managers:** Provides valuable insights for employers and managers to ensure a safer and more comfortable work environment for employees.

    **Use Cases:**
    - **Workplace Ergonomics:** Apply the tool to evaluate and improve ergonomics in various workplaces, including offices, manufacturing facilities, and healthcare settings.
    - **Ergonomic Training:** Use the tool as part of ergonomic training programs to educate employees about proper posture and ergonomic practices.
    - **Research and Development:** Support research and development efforts focused on ergonomic improvements and interventions.

    **Conclusion:**
    The ergonomic analysis tool is an essential resource for enhancing workplace safety and comfort through comprehensive posture analysis and actionable insights.
    """)


# Set up the page
st.set_page_config(page_title="Ergonomic Analysis using RULA and REBA", layout="centered")
page = st.sidebar.selectbox("Select a page", ["Home", "RULA", "REBA", "Purpose"])

if page == "Home":
    home_page()
elif page == "RULA":
    details_page1()
elif page == "REBA":
    details_page2()
elif page == "Purpose":
    purpose_page()

# Sidebar for file upload and buttons
if page == "Home":
    st.sidebar.title("Upload an Image or Video")
    uploaded_file = st.sidebar.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi"])

    # Live Webcam Analysis Buttons
    if st.sidebar.button("START LIVE POSTURE ANALYSIS"):
        st.session_state.video_processing_active = True
        st.session_state.video_processing_done = False
        video_pose_estimation(0, process_interval=20)

    if st.sidebar.button("STOP LIVE POSTURE ANALYSIS"):
        st.session_state.video_processing_active = False
        st.session_state.video_processing_done = True

    # Only show the Generate Report button if video processing is done
    if st.session_state.video_processing_done:
        if st.sidebar.button("GENERATE REPORT"):
            generate_report()

    # Handling file input
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        st.write(file_details)

        # Get MIME type
        mimestart = mimetypes.guess_type(uploaded_file.name)[0]

        if mimestart is not None:
            mimestart = mimestart.split('/')[0]
            if mimestart == 'video':
                with open("temp_video.mp4", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                video_pose_estimation("temp_video.mp4", process_interval=20)
            elif mimestart == 'image':
                with open("temp_image.jpg", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                image_pose_estimation("temp_image.jpg")
