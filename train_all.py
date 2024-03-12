from train_dense_labeling import main
config = {
    'batch_size': 128,
    'epochs':200,
    'fsl': False,
    'model': 'segmenter',
    'seed': 73054772,
    'dataset': 'physiq'
}
main(config)

config = {
    'batch_size': 128,
    'epochs':200,
    'fsl': False,
    'model': 'transformer',
    'seed': 73054772,
    'dataset': 'physiq'
}
main(config)

config = {
    'batch_size': 128,
    'epochs':200,
    'fsl': False,
    'model': 'LSTM',
    'seed': 73054772,
    'dataset': 'physiq'
}
main(config)

config = {
    'batch_size': 128,
    'epochs':200,
    'fsl': False,
    'model': 'CRNN',
    'seed': 73054772,
    'dataset': 'physiq'
}
main(config)

config = {
    'batch_size': 128,
    'epochs':200,
    'fsl': False,
    'model': 'CCRNN',
    'seed': 73054772,
    'dataset': 'physiq'
}
main(config)

config = {
    'batch_size': 128,
    'epochs':200,
    'fsl': False,
    'model': 'UNet',
    'seed': 73054772,
    'dataset': 'physiq'
}

# config = {
#     'batch_size': 128,
#     'epochs':200,
#     'fsl': False,
#     'model': 'UNet2',
#     'seed': 73054772,
#     'dataset': 'physiq'
# }
main(config)