
# the new config inherits the base configs to highlight the necessary modification
_base_ = 'C:/Users/guhan/OneDrive/Desktop/Cap stone/mmseg/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth' 
# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ("person","rider","motorcycle","bicycle","autorickshaw","car","truck","bus","vehicle fallback")
num_things_classes = 9
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
model = dict(
    panoptic_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes),
    test_cfg=dict(panoptic_on=False))

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
        ann_file=r'C:/Users/guhan/OneDrive/Desktop/Cap stone/IDD-data/IDD_Segmentation1/annotations/instancesonly_filtered_gtFine_test.json',
        img_prefix=r'C:/Users/guhan/OneDrive/Desktop/Cap stone/IDD-data/IDD_Segmentation1/leftImg8bit/test/')
    )


# 2. model settings

# # explicitly over-write all the `num_classes` field from default 80 to 5.
# model = dict(
#     roi_head=dict(
#         bbox_head=[
#             dict(
#                 type='Shared2FCBBoxHead',
#                 # explicitly over-write all the `num_classes` field from default 80 to 5.
#                 num_classes=4),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 # explicitly over-write all the `num_classes` field from default 80 to 5.
#                 num_classes=4),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 # explicitly over-write all the `num_classes` field from default 80 to 5.
#                 num_classes=4)],
#     # explicitly over-write all the `num_classes` field from default 80 to 5.
#     mask_head=dict(num_classes=4)))

# load_from = 'C:/Users/guhan/OneDrive/Desktop/Cap stone/mmseg/mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.38__segm_mAP-0.344_20200504_231812-0ebd1859.pth'
