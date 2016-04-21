from GRBM_DBN import test_GRBM_DBN
from load_data_MNIST import load_data

LAYER_SIZE = [256]
N_LAYERS = [2]
ITERATIONS = 1

datasets = load_data()
#
# for _ in range(ITERATIONS):
#     for n_layers in N_LAYERS:
#         for layer_size in LAYER_SIZE:
test_score, val_score = test_GRBM_DBN(finetune_lr=0.2, pretraining_epochs=[1, 1],
                pretrain_lr=[0.0001, 0.002], k=1, weight_decay=0.002,
                momentum=0.7, batch_size=20, datasets=datasets,
                hidden_layers_sizes=[784, 784], finetune = True,
                saveToDir = '../results/MNIST/', loadModelFromFile = None, verbose = True)

log = '../data/MNIST.log'
with open(log, 'a') as f:
    f.write('LAYER_SIZE=%d, n_layers=%d, test_score=%f%%, val_score=%f%%\n' % (layer_size, n_layers, test_score, val_score))

