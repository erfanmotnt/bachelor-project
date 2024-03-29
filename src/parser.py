import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset', 
					metavar='-d', 
					type=str, 
					required=True,
                    help="dataset from ['synthetic', 'SMD']")

parser.add_argument('--seed', 
					type=int, 
					required=True,
                    help="random seed")

parser.add_argument('--entity', 
					type=str, 
					required=True,
                    help="entity from dataset folder")

parser.add_argument('--model', 
					metavar='-m', 
					type=str, 
					required=False,
					default='LSTM_Multivariate',
                    help="model name")
parser.add_argument('--test', 
					action='store_true', 
					help="test the model")
parser.add_argument('--retrain', 
					action='store_true', 
					help="retrain the model")
parser.add_argument('--less', 
					action='store_true', 
					help="train using less data")
args = parser.parse_args()