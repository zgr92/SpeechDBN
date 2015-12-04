from GRBM_DBN import test_GRBM_DBN
from load_data_MNIST import load_data

datasets = load_data()

test_GRBM_DBN(finetune_lr=0.1, pretraining_epochs=[3, 1],
             pretrain_lr=[0.002, 0.02], k=1, weight_decay=0.0002,
             momentum=0.9, batch_size=128, datasets=datasets,
             hidden_layers_sizes=[512, 512, 512],
             filename='../data/MNIST.pickle')

