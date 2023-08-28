import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import dgl
from dgl.nn import GATConv
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from src.dlutils import *
from src.constants import *
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

torch.manual_seed(1)

## OmniAnomaly Model (KDD 19)
class OmniAnomaly(nn.Module):
	def __init__(self, feats):
		super(OmniAnomaly, self).__init__()
		self.name = 'OmniAnomaly'
		self.lr = 0.002
		self.beta = 0.01
		self.n_feats = feats
		self.n_hidden = 32
		self.n_latent = 8
		self.lstm = nn.GRU(feats, self.n_hidden, 2)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Flatten(),
			nn.Linear(self.n_hidden, 2*self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
		)

	def forward(self, x, hidden = None):
		hidden = torch.rand(2, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
		out, hidden = self.lstm(x.view(1, 1, -1), hidden)
		## Encode
		x = self.encoder(out)
		mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
		## Reparameterization trick
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		x = mu + eps*std
		## Decoder
		x = self.decoder(x)
		return x.view(-1), mu.view(-1), logvar.view(-1), hidden

# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(nn.Module):
	def __init__(self, feats):
		super(TranAD, self).__init__()
		self.name = 'TranAD'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_window = 10
		self.n = self.n_feats * self.n_window
		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
		decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		return tgt, memory

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
		# Phase 2 - With anomaly scores
		c = (x1 - src) ** 2
		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
		return x1, x2

class VAE_LSTM(nn.Module):
	def __init__(self, feats):
		super(VAE_LSTM, self).__init__()
		self.name = 'VAE_LSTM'
		self.lr = 0.001
		self.n_feats = feats
		self.n_window = 3
		self.n_hidden = 128
		self.n_latent = 16

		self.encoder_lstm = nn.LSTM(self.n_feats, self.n_hidden)
		self.sp1 =  nn.Softplus()
		self.z_mean = nn.Linear(self.n_hidden, self.n_latent)
		self.z_log_variance = nn.Linear(self.n_hidden, self.n_latent)
		self.decoder_lstm = nn.LSTM(self.n_latent, self.n_hidden)
		self.sp2 = nn.Softplus()
		self.x_mean = nn.Linear(self.n_hidden, self.n_feats)

	def get_z(self, mean, log_var):
		std = torch.exp(0.5 * log_var)
		noise = torch.randn(1, self.n_latent)
		z = noise * std + mean
		return z
		
	def forward(self, x):
		hidden = (torch.rand(1, 1, self.n_hidden, dtype=torch.float64), torch.randn(1, 1, self.n_hidden, dtype=torch.float64))
		hidden2 = (torch.rand(1, 1, self.n_hidden, dtype=torch.float64), torch.randn(1, 1, self.n_hidden, dtype=torch.float64))
		outputs = []
		loss_mats = []
		for _, g in enumerate(x):
			out, hidden = self.encoder_lstm(g.view(1, 1, -1), hidden)
			self.out = self.sp1(out)
			z_mean = self.z_mean(out).reshape(self.n_latent) ##pass hidden except out
			z_log_varaince = self.z_log_variance(out).reshape(self.n_latent)
			z = self.get_z(z_mean, z_log_varaince)
			out, hidden2 = self.decoder_lstm(z.view(1, 1, -1), hidden2)
			out = self.sp2(out)
			x_mean = self.x_mean(out).view(-1)
			outputs.append(x_mean)
			loss_mats.append((z_mean, z_log_varaince, z,
		     			 x_mean, g))
			
		return torch.stack(outputs), loss_mats
	

class DRNN(nn.Module):
	
	def __init__(self, feat):
		super(DRNN, self).__init__()
		self.name = 'DRNN'
		self.lr = 0.001
		self.n_input = feat
		self.n_layers = 4
		self.n_hidden = 128
		self.n_window = 3
		self.dilations = [2 ** i for i in range(self.n_layers)]

		layers = []
		n_inp = self.n_input
		n_out = self.n_hidden
		for i in range(self.n_layers):
			if i == self.n_layers-1:
				n_out = self.n_input
			layers.append(nn.LSTM(n_inp, n_out))
			n_inp = self.n_hidden
			
		self.cells = nn.Sequential(*layers)

	def forward(self, x):
		x_hats = []
		hidden = [None for _ in range(self.n_layers)]
		for j, g in enumerate(x):
			input = g.view(1, 1, -1)
			for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
				if j == 0:
					hidden[i] = [(torch.rand(1, 1, cell.hidden_size, dtype=torch.float64),
						torch.randn(1, 1, cell.hidden_size, dtype=torch.float64)) for _ in range(dilation)]
				k = j%dilation
				input, hidden[i][k] = cell(input, hidden[i][k])
			g_hat = input.view(-1)
			x_hats.append(g_hat)
		return torch.stack(x_hats)


class KNN():
	def __init__(self, feat):
		self.name = 'KNN'
		self.n_input = feat
		self.n_window = 10
		self.pca_dim = 10
		self.k = 5
		self.pca = PCA(n_components=self.pca_dim)
		self.classifier = KNeighborsClassifier(n_neighbors=self.k, metric='euclidean')

	def fit(self, data_windows, data):
		self.pca.fit(data_windows)
		data_reduced = self.pca.transform(data_windows)
		classes = np.ones(data_reduced.shape)
		self.classifier.fit(data_reduced, classes)

	def predict(self, data_windows, data):
		data_reduced = self.pca.transform(data_windows)
		distances = self.classifier.kneighbors(data_reduced, return_distance=True)[0]
		distances = np.power(distances, 2)
		scores = distances.mean(axis=1)
		return scores, None
	

from statsmodels.tsa.stattools import adfuller
def test_stationarity(data_col, signif=0.05):
    if not data_col.any():
        return "Non-Stationary"
    adf_test = adfuller(data_col, autolag='AIC')
    p_value = adf_test[1]
    if p_value <= signif:
        return "Stationary"
    else:
        return "Non-Stationary"
    
import statsmodels.tsa.api as stats_api

class VAR():
	def __init__(self, feat):
		self.name = 'VAR'
		self.n_input = feat
		self.n_window = 10
		self.diff_order = 1
		self.lag = 10
		
	def select_features(self, data):
		features = []
		print(data.shape)
		for i in range(data.shape[1]):
			print(i, end=' ')
			if test_stationarity(data[:, i]) == 'Stationary':
				features.append(i)
				print("is Stationary", end=' ')
		if len(features) == 1:
			features = [features[0], features[0]]
		return features
	
	def differencing(self, data):
		res = []
		for i in range(data.shape[1]):
			res.append(np.diff(data[:, i], self.diff_order))
		return np.array(res).T
	
	def fit(self, data_windows, data):
		out = self.differencing(data)
		self.featurs = self.select_features(out)
		out = np.take(out, self.featurs, axis=1)
		self.var = stats_api.VAR(out)
		self.fitted_var = self.var.fit(self.lag)

	def predict(self, data_windows, data):
		out = self.differencing(data)
		X = np.take(out, self.featurs, axis=1)

		y_pred = []
		for i in range(self.lag, X.shape[0]):
			y_pred.append(self.fitted_var.forecast(X[i-self.lag:i, :], 1)[0])
		y_pred = np.array(y_pred)
		y_pred = np.concatenate((X[:self.lag, :], y_pred))
		l = nn.MSELoss(reduction = 'none')
		y_pred = torch.tensor(y_pred).float()
		X = torch.tensor(X).float()
		MSE = l(y_pred, X)
		loss = MSE.detach().numpy()
		lossFinal = np.mean(loss, axis=1)
		lossFinal = np.concatenate(([0], lossFinal))
		return lossFinal, None

class DGHL():
	def __init__(self, feat):
		self.name = 'DGHL'
		self.n_input = feat
