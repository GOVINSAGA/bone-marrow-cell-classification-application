from ultralytics import YOLO

# Load your trained model
model = YOLO(r'C:\Users\yesag\runs\classify\train2\weights\best.pt')

# Define class_dict if necessary (ensure it matches model.names)
class_dict = {0: 'Badminton', 1: 'Cricket', 2: 'Karate', 3: 'Soccer', 4: 'Swimming', 5: 'Tennis', 6: 'Wrestling'}

# Test image path
test_image = r'D:\WORK TO DO\projects\uploads\tgame.jpg'

# Predict
results = model.predict(source=test_image, device='cuda')

# Extract predicted class
predicted_idx = results[0].probs.data.argmax().item()
predicted_class = class_dict.get(predicted_idx, "Unknown Class")
print(f"Predicted Class: {predicted_class}")
