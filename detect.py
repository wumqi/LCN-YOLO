from ultralytics import YOLO

if __name__ == '__main__':

model = YOLO('yolo11n.pt') 

results = model.predict(
source='path/to/image.jpg', 
save=True, 
show=True,
conf=0.25,
iou=0.5
)
