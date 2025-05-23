from ultralytics import YOLO
from PIL import Image

model = YOLO('yolov8m.pt')

dataset_config = './vehicle.yaml'

model.train(
    data=dataset_config,
    epochs=100,          # cukup untuk fine-tuning, bisa ditambah jika perlu
    imgsz=512,           # 640 adalah kompromi bagus antara akurasi dan kecepatan
    batch=8,            # Tesla T4 biasanya cukup kuat untuk batch 16, kalau VRAM 16GB masih aman
    device='cuda',
    lr0=0.001,           # learning rate awal, aman untuk fine-tuning
    lrf=0.1,
    optimizer='SGD',
    workers=4,           # Google Colab kadang worker lebih dari 2 malah kurang stabil
    augment=True,
    val=True
)


model.val()