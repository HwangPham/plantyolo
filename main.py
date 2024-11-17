from ultralytics import YOLO
# training
model = YOLO("ultralytics/ultralytics/backbone/EfficientNet.yaml")
results = model.train(data="dataset/pomegranate_data", epochs=100, imgsz=640)