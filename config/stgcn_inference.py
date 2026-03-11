# ST-GCN конфиг для инференса на твоих данных

# модель
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_with_res=True,
        graph_cfg=dict(layout='coco', mode='spatial'),
        in_channels=3),
    cls_head=dict(
        type='GCNHead',
        num_classes=60,
        in_channels=256))

# датасет (минимальный для инференса)
dataset_type = 'PoseDataset'
data_root = ''
data_root_val = ''

# пайплайн обработки
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

data = dict(
    test=dict(
        type=dataset_type,
        ann_file=None,  # будет передан в инференсе
        pipeline=test_pipeline))

# метрики
evaluation = dict(interval=1, metrics=['top_k_accuracy'])

# чекпоинт
load_from = 'models/stgcn_ntu60_2d.pth'