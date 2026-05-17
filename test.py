
from datasets.div2k_dataset import DIV2KDataset
ds = DIV2KDataset('data/DIV2K/DIV2K_train_HR', 'data/DIV2K/DIV2K_train_LR_bicubic/X2', max_images=5)
lr, hr = ds[0]
print('LR shape:', lr.shape)
print('HR shape:', hr.shape)
print('Dataset is working!')