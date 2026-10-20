"""Selected before constructing any dataset/model; included in run manifests."""
ARCHITECTURE = 'mlp'
PATCH_SIZE = 4
LR_SELECTION = None

def configure(architecture, patch_size):
    global ARCHITECTURE, PATCH_SIZE
    if architecture not in ('mlp', 'cnn') or patch_size not in (2,4,8,16):
        raise ValueError('Unsupported architecture/patch size')
    ARCHITECTURE, PATCH_SIZE = architecture, patch_size

def specification():
    return {'dataset':'CIFAR100','architecture':ARCHITECTURE,'patch_size':PATCH_SIZE,
            'lr_selection_sha256': LR_SELECTION,
            'patch_grid':32//PATCH_SIZE,'patch_semantics':'shared RGB permutation; within-patch pixels unchanged',
            'mlp':[3072,300,300,100],
            'cnn':'Conv3x3(3,32)-ReLU-Pool2; Conv3x3(32,64)-ReLU-Pool2; Conv3x3(64,128)-ReLU-Pool2; Flatten-Linear(2048,256)-ReLU-Linear(256,100)',
            'augmentation':'none','batchnorm':False,'dropout':False,
            'reference_policy':'complete same-protocol bank; no historical CIFAR targets'}
