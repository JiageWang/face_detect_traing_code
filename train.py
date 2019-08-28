import cv2
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from augmentations import SSDAugmentation
from torch.utils.data import DataLoader
from widerface import WIDERFaceDetection
from model import ConvNet
from loss import MultiBranchLoss


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    images = [sample[0].unsqueeze(0) for sample in batch]
    images = torch.cat(images, dim=0)
    labels = [sample[1] for sample in batch]
    return images, labels


WIDERFace_ROOT = "F:\Datasets\人脸识别\WIDERFACE"
dataset = WIDERFaceDetection(WIDERFace_ROOT, transform=SSDAugmentation(640, (127.5, 127.5, 127.5)))
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

writer = SummaryWriter('logs')

model = ConvNet()
if torch.cuda.is_available():
    model = model.cuda()
    summary(model, input_size=(3, 640, 640), device='cuda')
else:
    summary(model, input_size=(3, 640, 640), device='cpu')

criterion = MultiBranchLoss(input_size=(640, 640), writer=writer)
optimizer = Adam(model.parameters(), lr=1e-5)

print(len(dataloader))
for i, (images, labels) in enumerate(dataloader):
    model.train()
    print("epoch {} batch {}:".format(0, i))
    optimizer.zero_grad()
    img = images[0].permute((1, 2, 0)).numpy().astype(np.uint8)
    if torch.cuda.is_available():
        images = images.cuda()
    outputs = model(images)
    loss = criterion(outputs, labels, i)
    writer.add_scalar('total_loss', loss.item(), global_step=i)
    print('total_loss', loss.item())
    loss.backward()
    optimizer.step()

    for name, layer in model.named_parameters():
        # writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(),
        #                      global_step=i)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(),
                             global_step=i)
