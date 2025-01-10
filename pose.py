import cv2
import mediapipe as mp

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load an image
image_path = "run.jpg"  # Replace with your image path
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not read the image from {image_path}")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(image_rgb)

    # Check if pose landmarks were found
    if results.pose_landmarks:
        print("Pose landmarks detected!")

        # Extract landmark data and print it
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            print(f"Landmark {idx}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")

        for landmark in results.pose_landmarks.landmark:
            # Get image dimensions
            h, w, c = image.shape

            # Convert normalized coordinates to pixel coordinates
            cx, cy = int(landmark.x * w), int(landmark.y * h)

            # Draw the keypoints
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)  # Green color, filled circle

        # Optional: Draw landmarks on the image with connections
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the output image
        cv2.imshow("Pose Landmarks", image)  # Image with keypoints only
        cv2.imshow("Pose drawing", annotated_image)  # Image with keypoints and connections
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No pose landmarks detected.")

# Release resources
pose.close()
