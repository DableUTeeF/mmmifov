_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101), roi_head=dict(bbox_head=dict(num_classes=1)))
classes = ('person', )
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))
