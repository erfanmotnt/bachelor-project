import os
models = ['KNN', 'OmniAnomaly', 'TranAD', 'VAR', 'VAE_LSTM', 'DRNN']
datasets = ["SMD", "SMAP", "MSL", "UCR"]#["SWaT", "WADI"]

root = 'processed/'
import random
for dataset in datasets:
    all_files = os.listdir(root + dataset)
    entities = []
    for file in all_files:
        if file[-9:] == 'train.npy':
            entities.append(file[:-10])
    sample_entities = random.sample(entities, 5)
    for entity in sample_entities:
        for model in models:
            for i in range(3):
                os.system(f"python3 main.py --model {model} --dataset {dataset} --retrain --entity {entity} --seed {i}")
