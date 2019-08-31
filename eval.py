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
    net = ConvNet().eval()
    net.load_state_dict(torch.load('epoch70.pth'))
    if torch.cuda.is_available():
        net = net.cuda()
    summary(net.cuda(), input_size=(3, 640, 640), batch_size=1, device='cuda')

    WIDERFace_ROOT = r"F:\Datasets\人脸识别\WIDERFACE"
    dataset = WIDERFaceDetection(WIDERFace_ROOT)

    writer = SummaryWriter('eval_log')

    img = dataset.pull_image(51)
    print(img)
    img = cv2.resize(img, (640, 640))
    cv2.imshow('img', img)
    cv2.waitKey()

    img = ToTensor()(img).unsqueeze(0)
    if torch.cuda.is_available():
        img = img.cuda()

    outs = net(img)
    for i, out in enumerate(outs):
        pre = torch.sigmoid(out.squeeze(0).permute((1, 2, 0)).cpu().detach()[:, :, 4].unsqueeze(2)).numpy()
        print("feature map {} max".format(i), np.max(pre))
        print("feature map {} min".format(i), np.min(pre))
        writer.add_histogram('out{}'.format(i), pre)
        writer.add_image('out{}'.format(i), pre, dataformats='HWC')
        cv2.imshow(str(i), pre)

    cv2.waitKey()
