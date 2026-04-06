import torch
import cv2
from torchvision import models, transforms

# load model
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 31)

model.load_state_dict(torch.load("../model/MysoraBestModel.pth", map_location="cpu"))
model.eval()

# preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# classes
classes = [
"Alef","Beh","Teh","Theh","Jeem","Hah","Khah","Dal","Thal","Reh",
"Zain","Seen","Sheen","Sad","Dad","Tah","Zah","Ain","Ghain",
"Feh","Qaf","Kaf","Lam","Meem","Noon","Heh","Waw","Yeh"
]

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    img = transform(frame).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output,1).item()

    label = classes[pred]

    cv2.putText(frame,label,(20,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),2)

    cv2.imshow("Prediction",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()