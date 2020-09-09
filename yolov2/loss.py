import torch
import numpy

def objectness_darknet(YOLOoutput, num_class=80):
    # yolov3  # (bs, anchors, grid, grid, classes + xywh) of three scales
    if isinstance(YOLOoutput, tuple):
        output_allscales = []
        weights = [1, 5, 10.0]
        for idx, (prob, weight) in enumerate(zip(YOLOoutput, weights)):
            YOLOoutput = prob.permute(0, 1, 4, 2, 3)
            bs, anchors, probs, grid, grid = YOLOoutput.size()
            output = YOLOoutput.view(bs, anchors, probs, -1)

            det_loss = (output[:, :, 4, :] + 1).clamp(min=0).mean(dim=1)

            det_loss = det_loss ** 2
            det_loss = weight*det_loss.mean(1)

            output_allscales.append(det_loss.unsqueeze(1))
        output_allscales = torch.cat(output_allscales, dim=1)
        output_allscales = output_allscales.mean(1)

        return output_allscales
    else:

        if YOLOoutput.dim() == 3:
            YOLOoutput = YOLOoutput.unsqueeze(0)
        batch = YOLOoutput.size(0)
        assert (YOLOoutput.size(1) == (5 + num_class) * 5)
        h = YOLOoutput.size(2)
        w = YOLOoutput.size(3)

        # transform the output tensor from [batch, 425, 13, 13]
        output = YOLOoutput.view(batch, 5, 5 + num_class, h * w)  # [batch, 5, 85, h*w]
        # output_objectness = torch.sigmoid(output[:, :, 4, :])

        # Note: the scores for an image range from about -40 to +2.5.  When NMS happens, only boxes with obj scores
        # greater than 0 will be considered to contain objects.  See line 98 of utils.py.
        output_objectness, _ = output[:, :, 4, :].max(dim=1)
        det_loss = (output[:, :, 4, :] + 1).clamp(min=0).mean(dim=1)
        det_loss = det_loss ** 2
        det_loss = det_loss.mean(1)
        
        return  det_loss