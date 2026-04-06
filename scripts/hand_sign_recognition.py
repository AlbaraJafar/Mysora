import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# Set device to MPS if available, otherwise CPU
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')

# Load the pretrained ResNet50 model
num_classes = 31  # Number of classes
model = models.resnet50(weights=None)

# Modify the final layer to output 31 classes (as per your training)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Move the model to the appropriate device
model = model.to(device)

# Path to your checkpoint file
checkpoint_path = "C:\\Users\\baraj\\Downloads\\Mysora App\\Quran 2\\Quran\\model\\best.pth"
# Load the saved model from the checkpoint (best_checkpoint.pth)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Model loaded successfully from checkpoint '{checkpoint_path}'")

# Set the model to evaluation mode
model.eval()

# Define the transformation to match training preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load class names in English
class_names = [
    'Ain', 'Al', 'Alef', 'Beh', 'Dad', 'Dal', 'Feh', 'Ghain', 'Hah', 'Heh', 'Jeem', 'Kaf', 'Khah',
    'Laa', 'Lam', 'Meem', 'Noon', 'Qaf', 'Reh', 'Sad', 'Seen', 'Sheen', 'Tah', 'Teh', 'Teh_Marbuta',
    'Thal', 'Theh', 'Waw', 'Yeh', 'Zah', 'Zain'
]

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Load a default font for displaying text on the screen
fontpath = "/Library/Fonts/Arial.ttf"  # Adjust the font path for your system
font = ImageFont.truetype(fontpath, 64)

# Start video capture
cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

last_predicted_class = None  # Store the last predicted class

with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5) as hands:

    detect_next = True  # Flag to control detection
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = hands.process(rgb_frame)

        if detect_next and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box coordinates
                h, w, c = frame.shape
                x_min = w
                y_min = h
                x_max = y_max = 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                # Add some padding to the bounding box
                padding = 100
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)

                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Crop the hand region
                hand_img = frame[y_min:y_max, x_min:x_max]

                # Preprocess the image
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                hand_pil = transforms.ToPILImage()(hand_img)
                input_tensor = preprocess(hand_pil)
                input_tensor = input_tensor.unsqueeze(0).to(device)

                # Predict the class
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, preds = torch.max(outputs, 1)
                    last_predicted_class = class_names[preds.item()]  # Store the predicted class

                detect_next = False  # Pause detection until spacebar is pressed
                break  # Process only one hand

        # If there's a previous prediction, display it
        if last_predicted_class:
            # Convert frame to PIL Image for proper text rendering
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)

            # Draw the last predicted class on the image
            draw.text((50, 50), last_predicted_class, font=font, fill=(0, 255, 0))

            # Convert back to OpenCV image (BGR)
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Display the resulting frame
        cv2.imshow('Hand Sign Recognition', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break  # Quit the program
        elif key == ord(' '):
            detect_next = True  # Press spacebar to detect the next letter

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()