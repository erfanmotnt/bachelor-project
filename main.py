import pickle
import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *

from src.utils import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
from src.loss import VAE_Loss
from sklearn.metrics import roc_auc_score

# from beepy import beep

def convert_to_windows(data, model):
	windows = []; w_size = model.n_window
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
	return torch.stack(windows)

def load_dataset(dataset, entity):
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	for file in ['train', 'test', 'labels']:
		if dataset == 'SMD': file = entity + '_' + file
		if dataset == 'SMAP': file = entity + '_' + file
		if dataset == 'MSL': file = entity + '_' + file
		if dataset == 'UCR': file = entity + '_' + file
		loader.append(np.load(os.path.join(folder, f'{file}.npy')))
	# loader = [i[:, debug:debug+1] for i in loader]
	if args.less: loader[0] = cut_array(0.2, loader[0])
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	return train_loader, test_loader, labels

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dims):
	import src.models
	model_class = getattr(src.models, modelname)
	model = model_class(dims).double()
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
	if os.path.exists(fname) and (not args.retrain or args.test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, (optimizer, scheduler, epoch, accuracy_list)

def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):
	l = nn.MSELoss(reduction = 'mean' if training else 'none')
	feats = dataO.shape[1]
	if 'OmniAnomaly' in model.name:
		if training:
			mses, klds = [], []
			for i, d in enumerate(data):
				y_pred, mu, logvar, hidden = model(d, hidden if i else None)
				MSE = l(y_pred, d)
				KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
				loss = MSE + model.beta * KLD
				mses.append(torch.mean(MSE).item()); klds.append(model.beta * torch.mean(KLD).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			y_preds = []
			for i, d in enumerate(data):
				y_pred, _, _, hidden = model(d, hidden if i else None)
				y_preds.append(y_pred)
			y_pred = torch.stack(y_preds)
			MSE = l(y_pred, data)
			return MSE.detach().numpy(), y_pred.detach().numpy()
	elif 'TranAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
		bs = model.batch if training else len(data)
		dataloader = DataLoader(dataset, batch_size = bs)
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		if training:
			for d, _ in dataloader:
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				z = model(window, elem)
				l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
				if isinstance(z, tuple): z = z[1]
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			for d, _ in dataloader:
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, bs, feats)
				z = model(window, elem)
				if isinstance(z, tuple): z = z[1]
			loss = l(z, elem)[0]
			return loss.detach().numpy(), z.detach().numpy()[0]
	elif 'VAE_LSTM' in model.name:
		l = VAE_Loss(model.n_feats)
		y_pred, loss_mats = model(data)
		if training:
			loss = l(loss_mats)
			tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			loss = l.get_loss(loss_mats)
			return loss, y_pred.detach().numpy()
	elif "DRNN" in model.name:
		y_pred = model(data)
		loss = l(y_pred, data)
		if training:
			tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			l = nn.MSELoss(reduction = 'mean' if training else 'none')
			MSE = l(y_pred, data)
			return MSE.detach().numpy(), y_pred.detach().numpy()
	else:
		y_pred = model(data)
		loss = l(y_pred, data)
		if training:
			tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			return loss.detach().numpy(), y_pred.detach().numpy()

def load_custom_model(modelname, dims):
	if modelname in ['KNN', 'VAR', 'DGHL']:
		import src.models
		model_class = getattr(src.models, modelname)
		model = model_class(dims)
		args = ()
	else:
		model, args = load_model(modelname, dims)
	return model, args

def train_model(model_args, trainD, trainO, testO, labels):
	if model.name in ['KNN', 'VAR']:
		model.fit(trainD, trainO)
		return ()
	elif model.name in 'DGHL':
		from src.dghl.experiment import train_DGHL
		from src.dghl.utils import basic_mc
		mc = basic_mc(trainO.shape[1], random_seed=args.seed)
		root_dir = f'./dghlresults/DGHL-nomask-smdlabels/{args.dataset}'
		os.makedirs(name=root_dir, exist_ok=True)
		train_data = np.array(trainO[:, None, :])
		test_data = np.array(testO[:, None, :])
		train_mask = np.ones(train_data.shape)
		test_mask = np.ones(test_data.shape)
		test_labels = (np.sum(labels, axis=1) >= 1) + 0
		scores = train_DGHL(mc=mc, train_data=[train_data], test_data=[test_data],
                train_mask=[train_mask], test_labels=[test_labels], test_mask=[test_mask], entities=[args.entity], make_plots=False, root_dir=root_dir)
		return (scores)
	else:
		optimizer, scheduler, epoch, accuracy_list = model_args
		if model.name in ['DRNN', 'VAE_LSTM']:
			num_epochs = 15
		else:
			num_epochs = 5
		e = epoch + 1
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
			accuracy_list.append((lossT, lr))
		return (optimizer, scheduler)

if __name__ == '__main__':
	train_loader, test_loader, labels = load_dataset(args.dataset, args.entity)
	import torch
	import random
	import numpy as np
	torch.manual_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)
	model, model_args = load_custom_model(args.model, labels.shape[1])

	## Prepare data
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))
	trainO, testO = trainD, testD
	if model.name in ['TranAD', 'KNN']: 
		trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

	### Training phase
	if not args.test:
		print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
		start = time()
		train_args = train_model(model_args, trainD, trainO, testO, labels)
		print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)

	### Testing phase
	print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
	if model.name in ['KNN', 'VAR']:
		lossFinal, _ = model.predict(testD, testO)
	elif model.name in 'DGHL':
		lossFinal = train_args
	else: 
		optimizer, scheduler = train_args
		torch.zero_grad = True
		model.eval()
		loss, _ = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
		lossFinal = np.mean(loss, axis=1)
	print(f'{color.HEADER}Result {args.model} on {args.dataset}{color.ENDC}')
	labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
	result = roc_auc_score(labelsFinal, lossFinal)
	with open(f'results{args.seed}/{model.name} {args.dataset} {args.entity}.pkl', 'wb') as handle:
		pickle.dump({'roc-auc': result, 'scores': lossFinal},
	       handle, protocol=pickle.HIGHEST_PROTOCOL)

	print(result)
	print()
