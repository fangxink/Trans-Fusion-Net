import argparse
import json
from os.path import join

import numpy as np
from PIL import Image
import torch.nn.functional as F
iw, ih = (500, 374)
w, h = 473, 473
scale = min(w / iw, h / ih)
nw = int(iw * scale)
nh = int(ih * scale)
orininal_h = 500
orininal_w = 374
import os

# 设标签宽W，长H
num_class = 21
# --------------------------------------------#
#   区分的种类，和json_to_dataset里面的一样
# --------------------------------------------#
#name_class = ["background", "pterygium"]
name_class = ["_background_","road","sidewalk","building","wall","fence","pole","traffic light","traffic sign","vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle"]
def fast_hist(a, b, n):
    # --------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    # --------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    # --------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    # --------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)

def classPixelAccuracy(hist):
    # return each category pixel accuracy(A more accurate way to call it precision)
    # acc = (TP) / TP + FP
    classAcc = np.diag(hist) / hist.sum(axis=1)
    return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)



def compute_iou(inputs,labels):

    hist = np.zeros((num_class, num_class))
    total_pa=0
    for i in range(inputs.shape[0]):
        pr = inputs[i]
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
        pr = pr[int((473 - nh) // 2):int((473 - nh) // 2 + nh),int((473 - nw) // 2):int((473 - nw) // 2 + nw)]
        label=labels[i]
        image = Image.fromarray(np.uint8(pr)).resize((orininal_w, orininal_h), Image.NEAREST)
        image.save("./miou_pr_dir/" + 'temp' + ".png")

        image=Image.open("./miou_pr_dir/" + 'temp' + ".png")
        os.remove("./miou_pr_dir/" + 'temp' + ".png")
        pred = np.array(image)

        con=fast_hist(label.flatten(), pred.flatten(), num_class)
        hist += con
        cpa = classPixelAccuracy(con)
        total_pa += cpa[1]
    mIoUs = per_class_iu(hist)
    mPA = per_class_PA(hist)
    pyterpa=total_pa/inputs.shape[0]

    iou=round(mIoUs[1], 2)
    pa=round(mPA[1], 2)
    return iou,pa,pyterpa

