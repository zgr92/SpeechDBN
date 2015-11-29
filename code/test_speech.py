from GRBM_DBN import test_GRBM_DBN
from load_data_speech import load_data

N_FRAMES = 9

datasets = load_data(n_frames=N_FRAMES)

test_GRBM_DBN(finetune_lr=0.1, pretraining_epochs=[225, 75],
             pretrain_lr=[0.002, 0.02], k=1,
             datasets=datasets, batch_size=128,
             hidden_layers_sizes=[1024, 1024, 1024],
             n_ins=39*N_FRAMES, n_outs=120,
             filename='../data/speech.pickle')

