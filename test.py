import torch
from model import YoloV1
from utils import *
from dataset import VOCDataset
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from train import Compose


DEVICE = "mps" if torch.backends.mps.is_available else "cpu"
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
WEIGHT_DECAY = 0
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
NUM_WORKERS = 0
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "trained_model_1000_examples.pth.tar"

def run():

    transform = Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = VOCDataset(
        "data/100examples.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )


    model = YoloV1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    checkpoint = torch.load(LOAD_MODEL_FILE)
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


    # if LOAD_MODEL:
    #     print("here")
    #     load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    model.eval()

    for idx, (x, y) in enumerate(test_loader):
        x = x.to(DEVICE)
        for id in range(8):
            bboxes = cellboxes_to_boxes(model(x))
            bboxes = non_max_suppression(bboxes[id], iou_threshold=0.5, threshold=0.5, box_format='midpoint')
            print(bboxes)
            plot_image(x[id].permute(1, 2, 0).to('cpu'), bboxes)
            print(f'image no {id} tested successfully!!!')
        if idx == 1:
            break

if __name__ == '__main__':
    run()