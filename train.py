import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from time import sleep
from torch.utils.data import DataLoader
from model import YoloV1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss

seed = 47
torch.manual_seed(seed)

LEARNING_RATE = 1e-5
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# print(f"$$$$$$$$$$$$$$$$$$$$$  using {DEVICE} device $$$$$$$$$$$$$$$$$$$$")


BATCH_SIZE = 16
WEIGHT_DECAY = 0.1
LOAD_MODEL = False
LOAD_MODEL_FILE = 'trained_model_100_examples.pth.tar'
EPOCHS = 1000
NUM_WORKERS = 4
PIN_MEMORY = True
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

class Compose(object):
    def __init__(self, transform):
        self.transforms = transform
    
    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes

transform = Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(mean_loss=(sum(mean_loss)/len(mean_loss)))
    print(f'Mean Loss was {sum(mean_loss)/len(mean_loss)}')


def main():
    model = YoloV1(split_size=7, num_classes=20, num_boxes=2).to(DEVICE)
    # print(model)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    loss_fn = YoloLoss().to(DEVICE)

    checkpoint = torch.load(LOAD_MODEL_FILE)
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    # if LOAD_MODEL:
    #     load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
    
    train_dataset = VOCDataset(
        'data/100examples.csv',
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )

    test_dataset = VOCDataset(
        'data/8examples.csv',
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False
    )

    for epoch in range(EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn)

        if epoch%10 == 0:

            pred_boxes, target_boxes = get_bboxes(
                train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
            )

            mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format='midpoint'
            )
            print(f'Train MAP: {mean_avg_prec}')

            if mean_avg_prec > 0.9:
                print("came here")
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            sleep(1)

if __name__ == '__main__':
    main()
