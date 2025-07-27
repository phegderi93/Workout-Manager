import cv2
import mediapipe as mp
import numpy as np
import math



def calculate_angle(a, b, c, stage, counter):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians * 180.0 / math.pi)

    if angle > 180.0:
        angle = 360 - angle

    # Classify position based on elbow angle

    if angle > 160:
        stage = "up"

    if angle < 50 and stage == 'up':
        stage = "down"
        counter += 1

    return angle, stage, counter



def extract_landmarks(results):

    try:
        landmark = results.pose_landmarks.landmark
        shoulder = [landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y]
    except:
        pass

    return shoulder, elbow, wrist



def process_frame(pose, frame):

    # get shape of the frame
    height, width, _ = frame.shape


    # Conver frome from BGR to RGB as mediapipe uses RGB and opencv2 is in BGR

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # Set the image to not writeable for performance

    results = pose.process(image)  # Process the image and get the pose landmarks

    image.flags.writeable = True  # Set the image to writeable again

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV

    return image, results, height, width



def renderui(image, counter, stage, angle, width):
    # Define some colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 200, 0)
    RED = (0, 0, 255)
    BLUE = (245, 117, 25)

    angle_max = 175
    angle_min = 25


    # Draw Title and background

    cv2.rectangle(image, (int(width / 2) - 150, 0), (int(width / 2) + 250, 73), BLUE, -1)
    cv2.putText(image, 'AI Workout Manager', (int(width / 2) - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                WHITE, 2, cv2.LINE_AA)

    # Reps
    # Display REPS label
    cv2.putText(image, 'REPS', (15, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2, cv2.LINE_AA)  # Thicker font, black color

    # Display counter number
    cv2.putText(image, str(counter), (15, 60), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 0, 0), 3, cv2.LINE_AA)  # Bigger & bolder

    # progrss bar
    progress = ((angle - angle_min) / (angle_max - angle_min)) * 100
    cv2.rectangle(image, (50, 350), (50 + int(progress * 2), 370), GREEN, cv2.FILLED)
    cv2.rectangle(image, (50, 350), (250, 370), WHITE, 2)
    cv2.putText(image, f'{int(progress)}%', (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2, cv2.LINE_AA)






    return image



def run_pose_detection(mp_drawing, mp_pose,filename):
    stage = None
    counter = 0
    # Read the vidoe files

    cap = cv2.VideoCapture(filename)

    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process Each Frame

            image, results, height, width  = process_frame(pose, frame)

            # Extract landmarks

            shoulder, elbow, wrist = extract_landmarks(results)

            # Calculate the angle between shoulder, elbow and wrist to detect pushup

            angle, stage, counter = calculate_angle(shoulder, elbow, wrist, stage, counter)

            image = renderui(image, counter, stage, angle, width)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=5),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=5))

            cv2.imshow('AI Workout Manager', image)




            if cv2.waitKey(5) & 0xFF == ord('q'):
                break






if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    filename = "assets/pushup.mp4"

    run_pose_detection(mp_drawing, mp_pose, filename)








