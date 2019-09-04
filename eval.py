import cv2
import torch
from torchsummary import summary
from torchvision.transforms import ToTensor
import numpy as np
from tensorboardX import SummaryWriter
from model import ConvNet
from widerface import WIDERFaceDetection
from augmentations import SSDAugmentation

if __name__ == "__main__":
    net = ConvNet()
    net.load_state_dict(torch.load('no_gassuion_epoch240.pth'))
    net = net.eval()
    if torch.cuda.is_available():
        net = net.cuda()
    summary(net.cuda(), input_size=(3, 640, 640), batch_size=1, device='cuda')

    WIDERFace_ROOT = r"F:\Datasets\人脸识别\WIDERFACE"
    dataset = WIDERFaceDetection(WIDERFace_ROOT)

    writer = SummaryWriter('eval_log')

    # img = dataset.pull_image(1144)
    img = cv2.imread('2.jpg')
    # cv2.waitKey()
    # _, img = cv2.VideoCapture(0).read()
    # img = cv2.resize(img, (640, 640))

    src = img.copy()
    img = ToTensor()(img).unsqueeze(0)
    if torch.cuda.is_available():
        img = img.cuda()

    outs = net(img)
    outputs = []
    strides = [16, 32, 64, 128, 213, 320]
    for j, out in enumerate(outs):
        out = out.permute((0, 2, 3, 1)).contiguous().squeeze(0)
        out[:, :, 4] = torch.sigmoid(out[:, :, 4])
        # out = out[out[:, :, 4]>0.9, :]
        mask = torch.nonzero(out[:, :, 4] > 0.95)
        for i in range(mask.size(0)):
            x = mask[i, 1].item()
            y = mask[i, 0].item()
            bbox = [x, y, strides[j]]
            bbox.extend(out[y, x, :].detach().cpu().numpy().tolist())
            outputs.append(bbox)
    print(outputs)
    for output in outputs:
        x1 = (output[0] + output[3]) * output[2]
        y1 = (output[1] + output[4]) * output[2]
        w = (1 + output[5]) * output[2]
        h = (1 + output[6]) * output[2]
        x2 = int(x1 + w)
        y2 = int(y1 + h)
        y1 = int(y1)
        x1 = int(x1)
        cv2.rectangle(src, (x1, y1), (x2, y2), (0, 0, 255))
        # cv2.rectangle(src, (y1, x1), (y2, x2), (0, 0, 255))

    shapes = [40, 20, 10, 5, 3, 2]
    strides = [16, 32, 64, 128, 213, 320]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    # for k, stride in enumerate(strides):
    #     for i in range(int(640/stride)+1):
    #         for j in range(int(640/stride)+1):
    #             cv2.rectangle(src, (0*i, 0*j), (stride*i, stride*j), (0,255,0))
    cv2.imshow('img', src)
    cv2.waitKey()
