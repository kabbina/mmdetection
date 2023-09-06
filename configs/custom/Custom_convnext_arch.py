
# the new config inherits the base configs to highlight the necessary modification
_base_ = "C:/Users/guhan/OneDrive/Desktop/Cap stone/mmseg/mmdetection/configs/convnext/cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco.py"

# We also need to change the num_classes in head to match the dataset's annotationd

# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ("person","rider","motorcycle","bicycle","autorickshaw","car","truck","bus","vehicle fallback")
model = dict(
        roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=9
                ,
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
                num_classes=9
                ,
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
                num_classes=9
                ,
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
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=9
            ,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)))
)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file=r"C:/Users/guhan/OneDrive/Desktop/Cap stone/IDD-data/IDD_Segmentation1/annotations/instancesonly_filtered_gtFine_train.json",
        img_prefix=r'C:/Users/guhan/OneDrive/Desktop/Cap stone/IDD-data/IDD_Segmentation1/leftImg8bit/train/'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file=r'C:/Users/guhan/OneDrive/Desktop/Cap stone/IDD-data/IDD_Segmentation1/annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix=r'C:/Users/guhan/OneDrive/Desktop/Cap stone/IDD-data/IDD_Segmentation1/leftImg8bit/val/'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file=r'C:\Users\guhan\OneDrive\Desktop\Cap stone\IDD-data\IDD_Segmentation1\annotations\test.json',
        img_prefix=r'C:\Users\guhan\OneDrive\Desktop\Cap stone\data\idd_temporal_test_1/')
    )



# 2. model settings

# # explicitly over-write all the `num_classes` field from default 80 to 5.


load_from = 'C:/Users/guhan/OneDrive/Desktop/Cap stone/mmseg/mmdetection/work_dirs_convnext_arch_postESA/custom_config/latest.pth'
