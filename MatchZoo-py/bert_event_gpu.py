import torch
import matchzoo as mz
from pytorch_transformers import AdamW, WarmupLinearSchedule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

numbers_neg = 3
batch_size = 1
epoch = 20

ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=numbers_neg))
ranking_task.metrics = [
    # mz.metrics.MeanAveragePrecision(),
    mz.metrics.Precision(k=20),
    # mz.metrics.NormalizedDiscountedCumulativeGain(k=20)
]

print('data loading ...')
train_pack_raw = mz.datasets.event.load_data('train', task=ranking_task)
dev_pack_raw = mz.datasets.event.load_data('dev', task=ranking_task, filtered=True)
test_pack_raw = mz.datasets.event.load_data('test', task=ranking_task, filtered=True)
print('data loaded as `train_pack_raw` `dev_pack_raw` `test_pack_raw`')

preprocessor = mz.models.Bert.get_default_preprocessor()

train_pack_processed = preprocessor.transform(train_pack_raw)
valid_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)

trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=2,
    num_neg=numbers_neg,
    batch_size=batch_size,
    resample=True,
    sort=False
)
devset = mz.dataloader.Dataset(
    batch_size=batch_size,
    data_pack=valid_pack_processed
)
testset = mz.dataloader.Dataset(
    batch_size=batch_size,
    data_pack=test_pack_processed
)

padding_callback = mz.models.Bert.get_default_padding_callback(
    fixed_length_left=250,
    fixed_length_right=250
)

trainloader = mz.dataloader.DataLoader(
    device=device,
    dataset=trainset,
    stage='train',
    callback=padding_callback
)
devloader = mz.dataloader.DataLoader(
    device=device,
    dataset=devset,
    stage='dev',
    callback=padding_callback
)
testloader = mz.dataloader.DataLoader(
    device=device,
    dataset=testset,
    stage='test',
    callback=padding_callback
)

model = mz.models.Bert()

model.params['task'] = ranking_task
model.params['mode'] = 'bert-base-uncased' #预训练模型下载到/home/zhaolin/.cache/torch/pytorch_transformers
model.params['dropout_rate'] = 0.2

model.build()

print(model)
print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 5e-5},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, betas=(0.9, 0.98), eps=1e-8)
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=6, t_total=-1)

trainer = mz.trainers.Trainer(
    device=device,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    trainloader=trainloader,
    validloader=devloader,
    validate_interval=None,
    epochs=epoch,
    save_dir='./save/bert',
    save_all=True
)

trainer.run()
result = trainer.evaluate(testloader)
print(result)

# with open('bert_event_exp_50.log', 'a+') as f:
#     for i in result:
#         f.write(str(round(result[i], 3)) + '\t')
#     f.write('\n')