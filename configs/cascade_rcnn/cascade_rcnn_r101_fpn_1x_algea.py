_base_ = './cascade_rcnn_r50_fpn_1x_coco.py'
model = dict(pretrained='torchvision://resnet101',
             backbone=dict(depth=101),
             roi_head=dict(bbox_head=[
                 dict(
                     type='Shared2FCBBoxHead',
                     in_channels=256,
                     fc_out_channels=1024,
                     roi_feat_size=7,
                     num_classes=80,
                     bbox_coder=dict(
                         type='DeltaXYWHBBoxCoder',
                         target_means=[0., 0., 0., 0.],
                         target_stds=[0.1, 0.1, 0.2, 0.2]),
                     reg_class_agnostic=True,
                     loss_cls=dict(
                         type='CrossEntropyLoss',
                         use_sigmoid=False,
                         loss_weight=1.0),
                     loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                                    loss_weight=1.0)),
                 dict(
                     type='Shared2FCBBoxHead',
                     in_channels=256,
                     fc_out_channels=1024,
                     roi_feat_size=7,
                     num_classes=80,
                     bbox_coder=dict(
                         type='DeltaXYWHBBoxCoder',
                         target_means=[0., 0., 0., 0.],
                         target_stds=[0.05, 0.05, 0.1, 0.1]),
                     reg_class_agnostic=True,
                     loss_cls=dict(
                         type='CrossEntropyLoss',
                         use_sigmoid=False,
                         loss_weight=1.0),
                     loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                                    loss_weight=1.0)),
                 dict(
                     type='Shared2FCBBoxHead',
                     in_channels=256,
                     fc_out_channels=1024,
                     roi_feat_size=7,
                     num_classes=80,
                     bbox_coder=dict(
                         type='DeltaXYWHBBoxCoder',
                         target_means=[0., 0., 0., 0.],
                         target_stds=[0.033, 0.033, 0.067, 0.067]),
                     reg_class_agnostic=True,
                     loss_cls=dict(
                         type='CrossEntropyLoss',
                         use_sigmoid=False,
                         loss_weight=1.0),
                     loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
             ]))
lr_config = dict(step=[16, 19])
total_epochs = 20
