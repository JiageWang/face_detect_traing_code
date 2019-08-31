import cv2
import numpy as np
import torch
from torch.optim import Adam, SGD
from tensorboardX import SummaryWriter
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


epochs = 100
batch_size = 32
learing_rate = 0.0001
obj_scale = 0.52
nobj_scale = 0.56
loc_scale = 0.02
show_iter = 10

WIDERFace_ROOT = r"F:\Datasets\人脸识别\WIDERFACE"
dataset = WIDERFaceDetection(WIDERFace_ROOT, transform=SSDAugmentation(640, (127.5, 127.5, 127.5)))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

writer = SummaryWriter(
    'logs/bigmodel_batch_{}_lr_{}_obj_{}_nobj_{}_loc_{}'.format(batch_size, learing_rate, obj_scale, nobj_scale,
                                                                loc_scale))

model = ConvNet()
if torch.cuda.is_available():
    model = model.cuda()
    summary(model, input_size=(3, 640, 640), device='cuda')
else:
    summary(model, input_size=(3, 640, 640), device='cpu')
model.load_state_dict(torch.load('/home/jiage/Desktop/detect/epoch3.pth'))

criterion = MultiBranchLoss(input_size=(640, 640), writer=writer, obj_scale=obj_scale, nobj_scale=nobj_scale,
                            loc_scale=loc_scale)
optimizer = Adam(model.parameters(), lr=learing_rate)

batchs_loss = 0
for epoch in range(epochs):
    model.train()
    dataset = WIDERFaceDetection(WIDERFace_ROOT, transform=SSDAugmentation(640, (127.5, 127.5, 127.5)))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    for i, (images, labels) in enumerate(dataloader):
        batch_num = epoch * len(dataloader) + i + 1
        optimizer.zero_grad()
        if torch.cuda.is_available():
            images = images.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels, batch_num)
        batchs_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_num % show_iter == 0:
            average_loss = batchs_loss / show_iter
            print("epoch {} batch {}:".format(epoch, i))
            print('total_loss', average_loss)
            writer.add_scalar('total_loss', average_loss, global_step=batch_num)
            batchs_loss = 0

            for name, layer in model.named_parameters():
                writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(),
                                     global_step=batch_num)
                writer.add_histogram(name + '_data', layer.cpu().data.numpy(),
                                     global_step=batch_num)
    torch.save(model.state_dict(), "epoch{}.pth".format(epoch))
