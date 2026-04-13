from ultralytics import YOLO

if __name__ == '__main__':
    # 加载预训练模型
model = YOLO('yolo11n.pt') # 替换为你的模型路径

# 对单张图片进行预测
results = model.predict(
source='path/to/image.jpg', # 替换为你的图片路径
save=True, # 保存预测结果
show=True, # 显示预测结果
conf=0.25, # 置信度阈值
iou=0.5 # IoU阈值
)