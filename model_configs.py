# our models
def double_conv_1d(in_channels, out_channels):
    # text generation
    return ('conv1d', [out_channels, in_channels, 3, 1, 1]), \
            ('relu', [True]),\
            ('conv1d', [out_channels, out_channels, 3, 1, 1]),\
            ('relu', [True])
aspp_dim = [5, 25, 50, 100] # trainable param: 493k
EXACT_config = [
    ('tpe', [200, 6]),
    ('aspp', [12, 6, 3, aspp_dim]),  # Assuming dilations are given, CAREFUL! in ASPP nn module, it should be 6, 64...
    *double_conv_1d(12 * len(aspp_dim), 64),# 3.5m (3,500k) trainable params
    ('store_conv', []),
    ('max_pool1d', [2, 2, 0]),
    *double_conv_1d(64, 128),
    ('store_conv', []),
    ('max_pool1d', [2, 2, 0]),
    *double_conv_1d(128, 256),
    ('store_conv', []),
    ('max_pool1d', [2, 2, 0]),
    *double_conv_1d(256, 512),
    ('upsample_unet', [2]),
    *double_conv_1d(512 + 256, 256),
    ('upsample_unet', [2]),
    *double_conv_1d(256 + 128, 128),
    ('upsample_unet', [2]),
    *double_conv_1d(128 + 64, 64),
    ('conv1d', [1, 64, 1, 1, 0]),
    ('sigmoid', []),
    ('indicing', [200]) # output length
]


BASELINE_config = [ # maml
    ('conv1d', [32, 6, 5, 1, 0]),  # Use conv1d. 6 is the input channel size. Adjust other numbers as needed.
    ('relu', [True]),
    ('bn', [32]),
    ('dropout', [0.5, True]), # 0.5
    ('max_pool1d', [2, 2, 0]),
    ('conv1d', [64, 32, 5, 1, 0]),
    ('relu', [True]),
    ('bn', [64]),
    ('dropout', [0.5, True]), 
    ('max_pool1d', [2, 2, 0]),
    ('flatten', []),
    ('linear', [200, 64 * 47]),  # 64 is the last number of channels, and 21. You might need to adjust this depending on the exact size after pooling.
    ('sigmoid', [])
]

CNN_BASELINE_config = [
        ('tpe', [200, 6]),
        ('aspp', [12, 6, 3, aspp_dim]),  # Assuming dilations are given, CAREFUL! in ASPP nn module, it should be 6, 64...
        *double_conv_1d(12 * len(aspp_dim), 64),# 2.5m (2,500k) trainable params
        ('store_conv', []),
        ('max_pool1d', [2, 2, 0]),
        *double_conv_1d(64, 128),
        ('store_conv', []),
        ('max_pool1d', [2, 2, 0]),
        *double_conv_1d(128, 256),
        ('store_conv', []),
        ('max_pool1d', [2, 2, 0]),
        *double_conv_1d(256, 512),
        ('upsample_unet', [2]),
        *double_conv_1d(512 + 256, 256),
        ('upsample_unet', [2]),
        *double_conv_1d(256 + 128, 128),
        ('upsample_unet', [2]),
        *double_conv_1d(128 + 64, 64),
        ('conv1d', [1, 64, 1, 1, 0]),
        ('sigmoid', []),
        ('indicing', [50])#args.input_length])
    ]


WITHOUT_ASPP_config = [
        ('tpe', [200, 6]),  # Assuming dilations are given, CAREFUL! in ASPP nn module, it should be 6, 64...
        *double_conv_1d(6, 64),# 2.5m (2,500k) trainable params
        ('store_conv', []),
        ('max_pool1d', [2, 2, 0]),
        *double_conv_1d(64, 128),
        ('store_conv', []),
        ('max_pool1d', [2, 2, 0]),
        *double_conv_1d(128, 256),
        ('store_conv', []),
        ('max_pool1d', [2, 2, 0]),
        *double_conv_1d(256, 512),
        ('upsample_unet', [2]),
        *double_conv_1d(512 + 256, 256),
        ('upsample_unet', [2]),
        *double_conv_1d(256 + 128, 128),
        ('upsample_unet', [2]),
        *double_conv_1d(128 + 64, 64),
        ('conv1d', [1, 64, 1, 1, 0]),
        ('sigmoid', []),
        ('indicing', [200])#args.input_length])
    ]


WITHOUT_TPE_config = [
        ('aspp', [12, 6, 3, aspp_dim]),  # Assuming dilations are given, CAREFUL! in ASPP nn module, it should be 6, 64...
        *double_conv_1d(12 * len(aspp_dim), 64),# 3.5m (3,500k) trainable params
        ('store_conv', []),
        ('max_pool1d', [2, 2, 0]),
        *double_conv_1d(64, 128),
        ('store_conv', []),
        ('max_pool1d', [2, 2, 0]),
        *double_conv_1d(128, 256),
        ('store_conv', []),
        ('max_pool1d', [2, 2, 0]),
        *double_conv_1d(256, 512),
        ('upsample_unet', [2]),
        *double_conv_1d(512 + 256, 256),
        ('upsample_unet', [2]),
        *double_conv_1d(256 + 128, 128),
        ('upsample_unet', [2]),
        *double_conv_1d(128 + 64, 64),
        ('conv1d', [1, 64, 1, 1, 0]),
        ('sigmoid', []),
        ('indicing', [200])#args.input_length])
    ]