import torch
from torchtext import data


def build_bucket_iterator(dataset, device, batch_size, is_train):
    device_obj = None if device is None else torch.device(device)
    iterator = data.BucketIterator(
        dataset=dataset,
        batch_size=batch_size,
        repeat=False,
        sort_key=dataset.sort_key,
        sort=False,
        # sorts the data within each minibatch in decreasing order
        # set to true if you want use pack_padded_sequences
        sort_within_batch=is_train,
        # shuffle batches
        shuffle=is_train,
        device=device_obj,
        train=is_train,
    )
    return iterator
