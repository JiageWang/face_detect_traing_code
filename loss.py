import numpy as np
import torch
import torch.nn as nn

from utils.bbox import xxyy2xywh
from utils.gauss_map import get_gaussion_mask


class MultiBranchLoss(nn.Module):
    def __init__(self, input_size, writer=None, pre_thred=0.0045, obj_scale=2, nobj_scale=1.1, loc_scale=0.001):
        super().__init__()
        self.input_size = input_size
        self.loc_loss = nn.MSELoss()
        self.pre_loss = nn.BCELoss()
        self.writer = writer
        self.pre_thred = pre_thred
        self.obj_scale = obj_scale
        self.nobj_scale = nobj_scale
        self.loc_scale = loc_scale

    def forward(self, outputs, targets, step):
        targets = get_match_targets(outputs, targets, self.input_size)
        # 可视化置信度图与目标图

        targets = [target.permute((0, 2, 3, 1)).contiguous().view(target.size(0), -1, target.size(1)) for target in
                   targets]
        targets = torch.cat(targets, dim=1)
        targets = targets.view(-1, targets.size(2))
        targets = targets.cuda()

        outputs = [output.permute((0, 2, 3, 1)).contiguous().view(output.size(0), -1, output.size(1)) for output in
                   outputs]
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.view(-1, outputs.size(2))

        outputs[:, 4] = torch.sigmoid(outputs[:, 4])  # 归一化到0-1

        pos_mask = targets[:, 4] == 1
        neg_mask = targets[:, 4] == 0

        neg_pre_loss = self.pre_loss(outputs[neg_mask, 4], targets[neg_mask, 4])
        pos_pre_loss = self.pre_loss(outputs[pos_mask, 4], targets[pos_mask, 4])
        x_loss = self.loc_loss(outputs[pos_mask, 0], targets[pos_mask, 0])
        y_loss = self.loc_loss(outputs[pos_mask, 1], targets[pos_mask, 1])
        w_loss = self.loc_loss(outputs[pos_mask, 2], targets[pos_mask, 2])
        h_loss = self.loc_loss(outputs[pos_mask, 3], targets[pos_mask, 3])
        pos_loc_loss = x_loss + y_loss + w_loss + h_loss
        sum_loss = self.loc_scale * pos_loc_loss + self.obj_scale * pos_pre_loss + self.nobj_scale * neg_pre_loss
        if self.writer:
            self.writer.add_scalar('neg_pre_loss', self.nobj_scale * neg_pre_loss.item(), global_step=step)
            self.writer.add_scalar('pos_pre_loss', self.obj_scale * pos_pre_loss.item(), global_step=step)
            self.writer.add_scalar('pos_loc_loss', self.loc_scale * pos_loc_loss.item(), global_step=step)

        return sum_loss


def get_match_targets(outputs, targets, input_size):
    shapes = [output.detach().cpu().numpy().shape[2:] for output in outputs]
    strides = np.array([input_size[0] / shape[0] for shape in shapes])
    branch_targets = [torch.zeros_like(output.cpu()) for output in outputs]
    # 对于每个样本
    for j, labels in enumerate(targets):
        # 对于每个目标
        for i in range(labels.shape[0]):
            # 计算绝对坐标
            x1 = labels[i, 0] * input_size[0]
            y1 = labels[i, 1] * input_size[1]
            x2 = labels[i, 2] * input_size[0]
            y2 = labels[i, 3] * input_size[1]
            x, y, w, h = xxyy2xywh(x1, y1, x2, y2)
            area = w * h
            # 寻找对应的特征图
            branch_index = (np.abs(strides ** 2 - area)).argmin()
            stride = strides[branch_index]
            # 分配到对应坐标
            x = int(x / stride)  # x坐标前有多少个cell
            y = int(y / stride)
            # 换算到相对cell左上点坐标（图像坐标） x1,y1, w, h
            labels[i, 0] = x1/stride - x
            labels[i, 1] = y1/stride - y
            labels[i, 2] = w/stride - 1
            labels[i, 3] = h/stride - 1
            branch = branch_targets[branch_index]

            if branch[j, 4, y, x] != 1:
                branch[j, 4, y, x] = 1
                branch[j, :4, y, x] = torch.tensor(labels[i, :4], dtype=torch.float)
    return branch_targets
