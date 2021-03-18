_base_ = './retinanet_r50_fpn_1x_coco.py'
# learning policy
lr_config = dict(step=[16, 22])
total_epochs = 24
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2),
        mask_head=dict(num_classes=2)))
