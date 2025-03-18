import cv2
import torch
from torchvision import transforms, models
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np
import torch.nn as nn


# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiTaskModel(nn.Module):
    def __init__(self, pretrained=True):
        super(MultiTaskModel, self).__init__()

        # Backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Fully connected layer for age prediction
        self.shared_fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # Final prediction for age
        self.age_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.shared_fc(x)
        age = self.age_head(x)
        age = torch.clamp(age, 0, 100)
        return age

# Load the model and weights
model_path = "/Users/khoale/Desktop/Programming 1/best_model.pth"
model = MultiTaskModel(pretrained=False)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

def align_face(image, landmarks):
        if landmarks is None or len(landmarks) < 1 or len(landmarks[0]) < 2:
            return image,1
        # Extract landmarks for eyes
        left_eye = landmarks[0][0]  # Left eye (x, y)
        right_eye = landmarks[0][1]  # Right eye (x, y)

        # Compute the angle to align the eyes horizontally
        dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Compute center between the eyes
        eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

        # Compute affine transform for rotation
        rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale=1)

        # Apply rotation to align face
        aligned_image = cv2.warpAffine(
            np.array(image),
            rotation_matrix,
            (image.size[0], image.size[1]),
            flags=cv2.INTER_CUBIC
        )
        return aligned_image,0

def crop_align(image, use_mtcnn):
    boxes, _, landmarks = use_mtcnn.detect(image, landmarks=True)
    if landmarks is not None:
        image, check = align_face(image, landmarks)
        if check == 0:
            image = Image.fromarray(image)
    if boxes is not None and len(boxes) != 0:
        box = boxes[0]
        box = [int(coord) for coord in box]
        image = image.crop((box[0], box[1], box[2], box[3]))
    else:
        return None
    return image

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device)

# Function to preprocess a detected face
def preprocess_face(face_image):
    face = crop_align(face_image, mtcnn)
    face = transform(face_image)

    return face.unsqueeze(0).to(device)  # Add batch dimension

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Press 'q' to quit the live demo.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Detect faces
    boxes, probs = mtcnn.detect(pil_image)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # Crop and preprocess face
            face = pil_image.crop((x1, y1, x2, y2))
            try:
                input_tensor = preprocess_face(face)
                with torch.no_grad():
                    age_prediction = model(input_tensor)
                    predicted_age = age_prediction.item()

                # Draw bounding box and age prediction on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Age: {predicted_age:.1f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error processing face: {e}")

    # Display the resulting frame
    cv2.imshow('Live Age Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
