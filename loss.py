import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, SmoothL1Loss

from utils.bbox import xxyy2xywh
from utils.gauss_map import get_gaussion_mask


class BranchLoss(nn.Module):
    def __init__(self, writer=None):
        super().__init__()
        self.loss = 0
        self.loc_loss = nn.MSELoss()
        self.pre_loss = nn.BCELoss()
        self.writer = writer

    def forward(self, output, target, branch, step):
        output = output.cpu()
        target = target.permute((0, 2, 3, 1)).contiguous().view(-1, target.size(1))
        output = output.permute((0, 2, 3, 1)).contiguous().view(-1, output.size(1))

        output[:, 4] = torch.sigmoid(output[:, 4])  # 归一化到0-1

        target[target[:, 4] <= 0, 4] = -1
        pos_mask = target[:, 4] >= 1  # 训练正样本与0-0.8之间的负样本
        neg_mask = (target[:, 4] > 0) & (target[:, 4] < 0.8)
        target[(target[:, 4] < 1) & (target[:, 4] >= 0.8), 4] = -1

        # pos_num = (target[:, 4] == 1).sum().item()
        # neg_num = (target[:, 4] == 0).sum().item()
        pos_num = pos_mask.sum()
        neg_num = neg_mask.sum()

        if pos_num != 0:
            neg_pre_loss = self.pre_loss(output[neg_mask, 4], target[neg_mask, 4])
            pos_pre_loss = self.pre_loss(output[pos_mask, 4], target[pos_mask, 4])
            pos_loc_loss = self.loc_loss(output[pos_mask, :4], target[pos_mask, :4])
            sum_loss = neg_pre_loss + pos_pre_loss
            print('neg_pre_loss : {:5f}'.format(neg_pre_loss.item()), end=' | ')
            print('pos_pre_loss : {:5f}'.format(pos_pre_loss.item()), end=' | ')
            print('pos_loc_loss: {:5f}'.format(pos_loc_loss.item()), end=' | ')
            print('sum_loss: {:5f}'.format(sum_loss.item()))
            self.writer.add_scalar('branch_{}_neg_pre_loss'.format(branch), neg_pre_loss.item(), global_step=step)
            self.writer.add_scalar('branch_{}_pos_pre_loss'.format(branch), pos_pre_loss.item(), global_step=step)
            self.writer.add_scalar('branch_{}_pos_loc_loss'.format(branch), pos_loc_loss.item(), global_step=step)
        else:
            sum_loss = torch.tensor(0)
            print('neg_pre_loss : {:5f}'.format(0), end=' | ')
            print('pos_pre_loss : {:5f}'.format(0), end=' | ')
            print('pos_loc_loss: {:5f}'.format(0), end=' | ')
            print('sum_loss: {:5f}'.format(0))
        return sum_loss


class MultiBranchLoss(nn.Module):
    def __init__(self, input_size, writer=None):
        super().__init__()
        self.input_size = input_size
        self.branchloss = BranchLoss(writer)

    def forward(self, outputs, targets, step):
        loss = torch.tensor(0)
        matched_targets = get_match_targets(outputs, targets, self.input_size)
        for i, (output, target) in enumerate(zip(outputs, matched_targets)):
            loss = torch.add(loss, self.branchloss(output, target, i, step))
        return loss


def get_match_targets(outputs, targets, input_size):
    shapes = [output.detach().cpu().numpy().shape[2:] for output in outputs]
    # strides = np.array([8, 16, 32, 64, 128, 256])
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
            # radio = 1.2 * np.sqrt(area / (stride ** 2))  # 越低层半径应越大
            radio = 5
            # 换算到相对cell左上点坐标
            labels[i, 0] = labels[i, 0] - x * stride / input_size[0]
            labels[i, 1] = labels[i, 1] - y * stride / input_size[1]
            labels[i, 2] = labels[i, 2] - x * stride / input_size[0]
            labels[i, 3] = labels[i, 3] - y * stride / input_size[1]

            mask = get_gaussion_mask(*shapes[branch_index], x, y, radio)
            if branch_targets[branch_index][j, 4, y, x] != 1:
                branch_targets[branch_index][j, 4, :, :] += torch.tensor(mask, dtype=torch.float)
                branch_targets[branch_index][j, :4, y, x] = torch.tensor(labels[i, :4], dtype=torch.float)
    return branch_targets
