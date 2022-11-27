import torch
import torch.nn.functional as F  

def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()  #比较input中元素大于（这里是严格大于）threshold中对应元素，大于则为1，不大于则为0
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)  #返回所有元素的平均值
    return score

def iou(inputs,target,smooth = 1e-5,threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    temp_inputs = torch.gt(temp_inputs, threhold).float()  # 比较input中元素大于（这里是严格大于）threshold中对应元素，大于则为1，不大于则为0
    # _temp_inputs = torch.lt(temp_inputs, threhold).float()

    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp
    # tn = torch.sum(_temp_inputs,axis=[0,1]) - fn

    iou=(tp+smooth)/(tp+fp+fn+smooth)
    iou=torch.mean(iou)
    return iou

def acc(inputs,target,smooth = 1e-5,threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    tempinputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    temp_inputs = torch.gt(tempinputs, threhold).float()  # 比较input中元素大于（这里是严格大于）threshold中对应元素，大于则为1，不大于则为0
    _temp_inputs = torch.le(tempinputs, threhold).float()

    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp
    tn = torch.sum(_temp_inputs,axis=[0,1]) - fn

    acc=(tn+tp+smooth)/(tn+tp+fp+fn+smooth)
    acc=torch.mean(acc)
    return acc

