# LCN-YOLO

## 1. Description
This repository provides the official implementation of **LCN-YOLO**, a lightweight and context-aware object detection framework designed for UAV aerial imagery. The model is optimized for detecting small objects under complex backgrounds while maintaining high computational efficiency.

---

## 2. Dataset Information

This project uses the following public datasets:

- **VisDrone Dataset**  
  https://github.com/VisDrone  

- **AI-TOD Dataset**  
  https://github.com/jwwangchn/AI-TOD  

### Dataset Preparation
Download the datasets and organize them as follows:
datasets/
├── VisDrone/
│ ├── images/
│ ├── labels/
├── AI-TOD/
│ ├── images/
│ ├── labels/


---

## 3. Code Structure

The main components of LCN-YOLO are organized as follows:

- `models/backbone/` – LI-ODConv dynamic backbone
- `models/neck/` – multi-scale feature fusion module (LMDC)
- `models/head/` – detection head (CGMA)

---

## 4. Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.12  
- CUDA ≥ 11.3  

## 5. Usage Instructions
## 5.1 Training
python train.py --data visdrone.yaml --cfg lcn-yolo.yaml --epochs 200 --batch 8
## 5.2 Testing
python test.py --weights best.pt --data visdrone.yaml
