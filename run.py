import os
models = ['VAE_LSTM', 'DRNN']#['KNN', 'OmniAnomaly', 'TranAD', 'VAR', 'VAE_LSTM', 'DRNN']
datasets = ["SMD", "SMAP", "MSL", "UCR"]#["SWaT", "WADI"]#["SMD", "SMAP", "MSL", "UCR"]#, 
datasets.reverse()
root = 'processed/'

for model in models:
    for dataset in datasets:
        all_files = os.listdir(root + dataset)
        entities = []
        for file in all_files:
            if file[-9:] == 'train.npy':
                entities.append(file[:-10])
        for entity in entities:
            os.system(f"python3 main.py --model {model} --dataset {dataset} --retrain --entity {entity}")
