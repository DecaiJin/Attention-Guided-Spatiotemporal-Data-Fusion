from .rs_paired_img import ImageDataset
from torch.utils.data import DataLoader


def make_dataset(opt, is_val=False):
    shuffle = False if is_val else True
    batch_size = 1 if is_val else opt.batch_size
    if is_val:
        dataset = ImageDataset(opt.test_path, opt.max_value, False)
    else:
        dataset = ImageDataset(opt.input, opt.max_value, opt.is_train)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=opt.workers,
    )
    return dataloader
