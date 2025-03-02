# Import necessary libraries and modules
from scipy.spatial import distance  
from imutils import face_utils  
from pygame import mixer  
import imutils  
import dlib 
import cv2  

# Initialize the Pygame mixer module for audio playback
mixer.init()

# Loading the audio file 
mixer.music.load("C:/Users/patta/Downloads/music.wav")

# Function to compute the eye aspect ratio (EAR) given a set of eye landmarks
def eye_aspect_ratio(eye):
    # Compute the Euclidean distances between various landmark points
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    # Calculate the eye aspect ratio using the formula: EAR = (A + B) / (2 * C)
    ear = (A + B) / (2.0 * C)
    return ear
    
# Constants for drowsiness detection
thresh = 0.30  # Threshold value for detecting drowsiness based on eye aspect ratio
frame_check = 40  # Number of consecutive frames to check for drowsiness (doubled)

# Initialize face detector and facial landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("C:/Users/patta/Downloads/shape_predictor_68_face_landmarks.dat")

# Define indices for left and right eye landmarks in the 68-point facial landmarks model
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Initialize video capture from default camera
cap = cv2.VideoCapture(0)
# Initialize a flag variable to keep track of consecutive frames with drowsiness
flag = 0

# Main loop for processing video frames
while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()
    # Resize the frame for faster processing
    frame = imutils.resize(frame, width=450)
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    subjects = detect(gray, 0)
    # Loop over detected faces
    for subject in subjects:
        # Predict facial landmarks for the detected face
        shape = predict(gray, subject)
        # Convert the predicted landmarks to NumPy array format
        shape = face_utils.shape_to_np(shape)
        
        # Check if both eyes are visible
        if len(shape[lStart:lEnd]) == 6 and len(shape[rStart:rEnd]) == 6:
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            # Calculate eye aspect ratio for each eye
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            
            if leftEAR < rightEAR:
                eye = leftEye
                ear = leftEAR
            else:
                eye = rightEye
                ear = rightEAR
            
            # Compute convex hull around the eye for visualization
            eyeHull = cv2.convexHull(eye)
            # Draw contours around the eye on the frame
            cv2.drawContours(frame, [eyeHull], -1, (0, 255, 0), 1)
            
            # Check if the eye aspect ratio falls below the threshold
            if ear < thresh:
                # Increment the flag counter
                flag += 1
                # Check if the flag counter exceeds the frame check value
                if flag >= frame_check:
                    # Print a drowsiness alert message
                    print("Drowsiness detected!")
                    # Display alert text on the frame
                    cv2.putText(frame, "****************ALERT!****************", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "****************ALERT!****************", (10,325),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # Play the alert sound
                    mixer.music.play()
            else:
                # Reset the flag counter if eyes are sufficiently open
                flag = 0
        else:
            # At least one eye is not detected
            cv2.putText(frame, "Eye(s) not detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the processed frame
    cv2.imshow("Frame", frame)
    # Check for key press to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup: Close all OpenCV windows and release video capture object
cv2.destroyAllWindows()
cap.release()
