from GRBM_DBN import test_GRBM_DBN
from load_data_MNIST import load_data

LAYER_SIZE = [256, 512, 1024]
N_LAYERS = [2, 3, 4]
ITERATIONS = 5

datasets = load_data()

for _ in range(ITERATIONS):
    for n_layers in N_LAYERS:
        for layer_size in LAYER_SIZE:
            test_score, val_score = test_GRBM_DBN(finetune_lr=0.1, pretraining_epochs=[225, 75],
                pretrain_lr=[0.002, 0.02], k=1, weight_decay=0.0002,
                momentum=0.9, batch_size=128, datasets=datasets,
                hidden_layers_sizes=n_layers*[layer_size], load=False,
                filename=('../data/MNIST_%d_%d.pickle'%(layer_size, n_layers)))

            log = '../data/MNIST.log'
            with open(log, 'a') as f:
                f.write('LAYER_SIZE=%d, n_layers=%d, test_score=%f%%, val_score=%f%%\n' % (layer_size, n_layers, test_score, val_score))

