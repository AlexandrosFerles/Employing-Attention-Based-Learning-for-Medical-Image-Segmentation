import torch
from torch.utils.data.sampler import SubsetRandomSampler


def create_k_folds(dataset, indices, k, batch_size):

    data_loaders = []
    step = int(len(indices) / k)
    start_index = 0

    for _ in range(k):

        end_index = min(start_index + step, len(indices))

        train_fold = indices[0:start_index] + indices[end_index:]
        val_fold = indices[start_index: end_index]

        train_sampler = SubsetRandomSampler(train_fold)
        val_sampler = SubsetRandomSampler(val_fold)

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=0, pin_memory=False
        )

        val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, sampler=val_sampler,
            num_workers=0, pin_memory=False
        )

        data_loaders.append((train_loader, val_loader))

        start_index += step

    return data_loaders


def create_k_folds_with_test_set(dataset, indices, k, batch_size):

    data_loaders = []
    step = int(len(indices) / k)
    start_index = 0

    for _ in range(k-1):

        end_index = min(start_index + 2*step, len(indices))
        val_index = start_index + step

        train_fold = indices[0:start_index] + indices[end_index:]
        val_fold = indices[start_index: val_index]
        test_fold = indices[val_index:end_index]

        print('{} {} {}'.format(((0, start_index), (end_index)), (start_index, val_index), (val_index, end_index)))

        train_sampler = SubsetRandomSampler(train_fold)
        val_sampler = SubsetRandomSampler(val_fold)
        test_sampler = SubsetRandomSampler(test_fold)

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=0, pin_memory=False
        )

        val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, sampler=val_sampler,
            num_workers=0, pin_memory=False
        )

        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, sampler=test_sampler,
            num_workers=0, pin_memory=False
        )

        data_loaders.append((train_loader, val_loader, test_loader))

        start_index += step

    return data_loaders


def create_k_LUNA_folds(dataset_list, k, batch_size):
    from torch.utils.data import ConcatDataset

    start_index = 0
    step = int(len(dataset_list) / k)
    val_index = step

    train_dataset = ConcatDataset(dataset_list[step:])
    test_dataset = ConcatDataset(dataset_list[:step])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=0, pin_memory=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1,
        num_workers=0, pin_memory=False
    )

    data_loaders = [(train_loader, test_loader)]

    for _ in range(k - 1):
        end_index = min(start_index + 2 * step, len(dataset_list))

        train_dataset = ConcatDataset(dataset_list[0:val_index] + dataset_list[end_index:])
        test_dataset = ConcatDataset(dataset_list[val_index: val_index + step])

        val_index += step
        start_index += step

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size,
            num_workers=0, pin_memory=False
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1,
            num_workers=0, pin_memory=False
        )

        data_loaders.append((train_loader, test_loader))

    return data_loaders