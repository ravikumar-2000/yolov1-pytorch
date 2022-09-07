import torch
from model import YoloV1
from utils import *
from dataset import VOCDataset
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader


DEVICE = "mps" if torch.backends.mps.is_available else 'cpu'
LEARNING_RATE = 2e-5
BATCH_SIZE = 64
WEIGHT_DECAY = 0
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"

transform = transforms.Compose([
    transforms.Resize((448, 448)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


test_dataset = VOCDataset(
    "data/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    shuffle=True,
    drop_last=True,
)


model = YoloV1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)

optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

if LOAD_MODEL:
    load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

for idx, (x, y) in enumerate(test_loader):
    x = x.to(DEVICE)
    for idx in range(1):
        bboxes = cellboxes_to_boxes(model(x))
        bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.5, box_format='midpoint')
        plot_image(x[idx].permute(1, 2, 0).to('cpu'), bboxes)
        print(f'image no {idx} tested successfully!!!')
