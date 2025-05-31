import torch
import torchvision.models as models
from torchvision.models import DenseNet121_Weights
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import os
import json
from datetime import datetime
import cv2
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

PLANT_CLASSES = [
    "tomato",
    "potato",
    "pepper",
    "corn",
    "soybean",
    "rice",
    "wheat",
    "cotton",
    "grape",
    "apple"
]

DISEASE_CLASSES = [
    "healthy",
    "bacterial_blight",
    "leaf_spot",
    "rust",
    "powdery_mildew",
    "early_blight",
    "late_blight",
    "leaf_curl",
    "mosaic_virus"
]

class PlantDiseaseModel(nn.Module):
    def __init__(self):
        super(PlantDiseaseModel, self).__init__()
        self.base_model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        num_features = self.base_model.classifier.in_features

        # Feature extraction layers
        self.shared_features = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Classifiers now take 512 input features
        self.plant_classifier = nn.Linear(512, len(PLANT_CLASSES))
        self.disease_classifier = nn.Linear(512, len(DISEASE_CLASSES))

    def forward(self, x):
        features = self.base_model.features(x)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        shared = self.shared_features(features)
        plant_pred = self.plant_classifier(shared)
        disease_pred = self.disease_classifier(shared)
        return plant_pred, disease_pred

def save_model_weights(model, accuracy=None):
    """Save model weights with timestamp and accuracy"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'plant_model_weights_{timestamp}.pth'
        if accuracy:
            filename = f'plant_model_weights_{timestamp}_{accuracy:.2f}.pth'
        
        torch.save(model.state_dict(), filename)
        torch.save(model.state_dict(), 'plant_model_weights.pth')
        print(f"Model weights saved as {filename}")
        return True
    except Exception as e:
        print(f"Error saving model weights: {e}")
        return False

def train_model(model, train_loader, val_loader, num_epochs=10):
    """Train the model with the given data loaders"""
    criterion_plant = nn.CrossEntropyLoss()
    criterion_disease = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
    
        for inputs, (plant_labels, disease_labels) in train_loader:
            inputs = inputs.to(device)
            plant_labels = plant_labels.to(device)
            disease_labels = disease_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            plant_outputs, disease_outputs = model(inputs)
            
            # Calculate loss
            plant_loss = criterion_plant(plant_outputs, plant_labels)
            disease_loss = criterion_disease(disease_outputs, disease_labels)
            total_loss = plant_loss + disease_loss
            
            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
        
        # Validation
        model.eval()
        val_accuracy = evaluate_model(model, val_loader)
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_model_weights(model, val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Loss: {running_loss/len(train_loader):.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}')

def evaluate_model(model, data_loader):
    """Evaluate model performance on the given data loader"""
    correct_plant = 0
    correct_disease = 0
    total = 0
    
    with torch.no_grad():
        for inputs, (plant_labels, disease_labels) in data_loader:
            inputs = inputs.to(device)
            plant_labels = plant_labels.to(device)
            disease_labels = disease_labels.to(device)
            
            plant_outputs, disease_outputs = model(inputs)
            
            _, plant_predicted = torch.max(plant_outputs.data, 1)
            _, disease_predicted = torch.max(disease_outputs.data, 1)
            
            total += plant_labels.size(0)
            correct_plant += (plant_predicted == plant_labels).sum().item()
            correct_disease += (disease_predicted == disease_labels).sum().item()
    
    plant_accuracy = correct_plant / total
    disease_accuracy = correct_disease / total
    return (plant_accuracy + disease_accuracy) / 2

def analyze_leaf_health(image):
    """Analyze leaf health metrics"""
    img_array = np.array(image)
    hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Extract the hue and saturation channels
    hue = hsv_img[:, :, 0]
    saturation = hsv_img[:, :, 1]
    
    # Calculate green color metrics
    green_mask = (hue >= 35) & (hue <= 85)
    healthy_area = np.sum(green_mask)
    total_area = green_mask.size
    health_ratio = healthy_area / total_area
    
    # Calculate affected area
    affected_mask = (saturation > 50) & ~green_mask
    affected_area = np.sum(affected_mask)
    affected_ratio = affected_area / total_area
    
    return {
        "health_ratio": float(health_ratio),
        "affected_ratio": float(affected_ratio),
        "total_leaf_area": int(total_area)
    }

def detect_disease_severity(image):
    """Detect disease severity in the plant leaf"""
    img_array = np.array(image)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Define masks for different color ranges that might indicate disease
    yellow_brown_mask = cv2.inRange(hsv, (20, 50, 50), (35, 255, 255))
    dark_spots_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 30))
    
    # Combine masks
    disease_mask = cv2.bitwise_or(yellow_brown_mask, dark_spots_mask)
    
    # Calculate affected area percentage
    total_pixels = img_array.shape[0] * img_array.shape[1]
    affected_pixels = np.sum(disease_mask > 0)
    severity = (affected_pixels / total_pixels) * 100
    
    # Determine severity level
    if severity < 10:
        return "Mild", severity
    elif severity < 30:
        return "Moderate", severity
    else:
        return "Severe", severity

def get_treatment_recommendations(disease_name, severity_level):
    """Get treatment recommendations based on disease and severity"""
    recommendations = {
        "bacterial_blight": {
            "Mild": [
                "Apply copper-based bactericide",
                "Improve air circulation",
                "Monitor regularly"
            ],
            "Moderate": [
                "Remove infected leaves",
                "Apply copper-based bactericide",
                "Reduce irrigation"
            ],
            "Severe": [
                "Remove heavily infected plants",
                "Apply systemic bactericide",
                "Implement crop rotation"
            ]
        },
        "healthy": {
            "Mild": ["Continue regular maintenance"],
            "Moderate": ["Continue regular maintenance"],
            "Severe": ["Continue regular maintenance"]
        }
    }
    
    # Get default recommendations if disease not in dictionary
    disease_recs = recommendations.get(disease_name, recommendations["healthy"])
    return disease_recs.get(severity_level, ["Consult local agricultural expert"])

def load_model():
    """Load and initialize the enhanced model"""
    try:
        model = PlantDiseaseModel()
        model = model.to(device)
        
        try:
            weights_path = 'plant_model_weights.pth'
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path))
                print(f"Loaded saved model weights from {weights_path}")
            else:
                print("No saved weights found, using initialized model")
                save_model_weights(model)
        except Exception as e:
            print(f"Error loading weights: {e}")
            save_model_weights(model)
        
        model.eval()
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def identify_plant_features(image):
    """Identify basic features of the plant from the leaf image."""
    img_array = np.array(image)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    features = {
        "color_mean": np.mean(hsv[:, :, 0]),
        "saturation_mean": np.mean(hsv[:, :, 1]),
        "lightness_mean": np.mean(lab[:, :, 0])
    }
    return features

def predict_image(image, model):
    """Predict plant disease using provided model"""
    try:
        # image is already a PIL Image
        original_image = image.convert('RGB')
        
        # Get leaf health metrics
        health_metrics = analyze_leaf_health(original_image)
        
        # Get disease severity
        severity_level, severity_percentage = detect_disease_severity(original_image)
        
        # Get plant features
        plant_features = identify_plant_features(original_image)
        
        # Model prediction
        image_tensor = transform(original_image).unsqueeze(0).to(device)
        model.eval()
        
        with torch.no_grad():
            plant_pred, disease_pred = model(image_tensor)
            
            # Get plant prediction
            plant_probs = torch.nn.functional.softmax(plant_pred, dim=1)
            plant_class = torch.argmax(plant_probs, dim=1).item()
            plant_confidence = plant_probs[0][plant_class].item()
            
            # Get disease prediction
            disease_probs = torch.nn.functional.softmax(disease_pred, dim=1)
            disease_class = torch.argmax(disease_probs, dim=1).item()
            disease_confidence = disease_probs[0][disease_class].item()
        
        plant_name = PLANT_CLASSES[plant_class]
        disease_name = DISEASE_CLASSES[disease_class]
        
        # Generate report
        report = {
            "PlantID": "UploadedImage",
            "Plant": {
                "Name": plant_name,
                "Confidence": round(float(plant_confidence) * 100, 2),  # as percentage
                "Features": plant_features
            },
            "Analysis": {
                "Disease": disease_name,
                "Confidence": round(float(disease_confidence) * 100, 2),  # as percentage
                "Severity": {
                    "Level": severity_level,
                    "Percentage": float(severity_percentage)
                },
                "LeafHealth": health_metrics
            },
            "Recommendations": get_treatment_recommendations(disease_name, severity_level),
            "TimeStamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return report, plant_class, disease_class, disease_name  # return predictions for accuracy
        
    except Exception as e:
        print(f"Error processing {image}: {str(e)}")
        return None, None, None, None

if __name__ == "__main__":
    folder_path = r"C:\Users\User\Desktop\leaf"
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    
    try:
        # Load the model
        model = load_model()
        
        # Get list of all valid images
        valid_images = [f for f in os.listdir(folder_path) 
                       if f.lower().endswith(image_extensions)]
        
        if not valid_images:
            print(f"No valid images found in {folder_path}")
            exit(1)
        
        # For accuracy calculation (if you have ground truth labels)
        total = 0
        correct_disease = 0
        correct_plant = 0

        # If you have a mapping of image names to true labels, load here:
        # Example: {"Test_1.jpg": {"plant": 0, "disease": 1}}
        # ground_truth = json.load(open("ground_truth.json"))
        ground_truth = {}  # <-- Fill this if you have labels

        # Process all images in the folder
        for image_name in valid_images:
            image_path = os.path.join(folder_path, image_name)
            print(f"\nProcessing image: {image_name}")
            
            # Process image
            prediction, plant_class, disease_class, disease_name = predict_image(image_path, model)
            
            if prediction:
                # Save result
                output_file = f"prediction_{os.path.splitext(image_name)[0]}.json"
                with open(output_file, 'w') as f:
                    json.dump(prediction, f, indent=2)
                
                print("\nPrediction result:")
                print(json.dumps(prediction, indent=2))
                print(f"\nResult saved to {output_file}")

                # Accuracy calculation if ground truth is available
                if image_name in ground_truth:
                    total += 1
                    if plant_class == ground_truth[image_name]["plant"]:
                        correct_plant += 1
                    if disease_class == ground_truth[image_name]["disease"]:
                        correct_disease += 1

        # Print accuracy if ground truth is available
        if total > 0:
            print(f"\nPlant Accuracy: {correct_plant/total*100:.2f}%")
            print(f"Disease Accuracy: {correct_disease/total*100:.2f}%")
        else:
            print("\nNo ground truth labels provided for accuracy calculation.")

    except Exception as e:
        print(f"Error: {str(e)}")