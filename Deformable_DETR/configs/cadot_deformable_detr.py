# ==========================================================
# CADOT – Deformable DETR (refine + two-stage)
# ==========================================================

# Base config provided by MMDetection (advanced Deformable DETR)
_base_ = [
    'mmdet::deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_coco.py'
]




data_root = "data_aug/"

metainfo = dict(
    classes=(
        "basketball field",
        "building",
        "crosswalk",
        "football field",
        "graveyard",
        "large vehicle",
        "medium vehicle",
        "playground",
        "roundabout",
        "ship",
        "small vehicle",
        "swimming pool",
        "tennis court",
        "train"
    )
)

# ----------------------------------------------------------
# DATA PIPELINES
# ----------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs')
]

# ----------------------------------------------------------
# DATALOADERS (CRITICAL FIX)
# ----------------------------------------------------------
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/train_split_fixed.json',
        data_prefix=dict(img='train/images/'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/val_split_fixed.json',
        data_prefix=dict(img='train/images/'),
        pipeline=val_pipeline
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test_valid/valid_as_test.json',
        data_prefix=dict(img='test_valid/images/'),
        pipeline=val_pipeline
    )
)

# ----------------------------------------------------------
# EVALUATORS
# ----------------------------------------------------------
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'train/val_split_fixed.json',
    metric='bbox',
    classwise=True
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test_valid/valid_as_test.json',
    metric='bbox',
    classwise=True
)





# ----------------------------------------------------------
# 2. MODEL
# ----------------------------------------------------------

model = dict(
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0),
                dict(type='IoUCost', iou_mode='giou', weight=2.0),
            ]
        )
    ),

    bbox_head=dict(
        num_classes=14,

        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0
        ),

        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
    )
)



# ----------------------------------------------------------
# 3. TRAINING
# ----------------------------------------------------------
train_cfg = dict(
    max_epochs=80
)

# ----------------------------------------------------------
# 4. OPTIMIZATION
# ----------------------------------------------------------

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=1e-4
    ),
    accumulative_counts=4,
    clip_grad=dict(max_norm=0.1, norm_type=2)
)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=1500
    )
]

# ----------------------------------------------------------
# 5. LOGGING
# ----------------------------------------------------------
default_hooks = dict(
    checkpoint=dict(interval=5),
    logger=dict(interval=50),
)

randomness = dict(seed=42)
