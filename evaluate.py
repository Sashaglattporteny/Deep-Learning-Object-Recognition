from ultralytics import YOLO
from roboflow import Roboflow

rf = Roboflow(api_key="5Gtyuy6iNi3PDjaEYyuK")
project = rf.workspace("ie-l0fg6").project("deep-ncu0q")
version = project.version(1)
dataset = version.download("yolov8")

# Load a model
model = YOLO("yolov8n.pt")

# Customize validation settings
validation_results = model.val(data="data.yaml")
