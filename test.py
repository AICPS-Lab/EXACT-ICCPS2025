from utils_loader import get_dataloaders


config = {
    'fg_label': [1, 2,3,4],
    'batch_size': 1,
    'n_way': 2,
    'n_shot': 4,
    'n_query': 2,
    'n_tasks_per_epoch': 500,
    'align': True,
}
train_loader, test_loader = get_dataloaders(config)

next(iter(train_loader))[0]
next(iter(train_loader))[0]