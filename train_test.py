import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ultralytics import YOLO

model = YOLO(model=r'C:\Users\11820\PycharmProjects\try\ultralytics-main\ultralytics\cfg\models\11\yolo11.yaml')
model.train(data='AI-TOD.yaml',
            cache=False,
            imgsz=640,
            epochs=300,
            batch=4,
            close_mosaic=0,
            resume=False,
            workers=0,
            optimizer='SGD',
            patience=0,
            project='runs/Attention',
            name='AI-TOD',
            )
