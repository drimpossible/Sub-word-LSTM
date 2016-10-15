#Author: Ameya Prabhu
import scipy.io
import h5py
import pickle
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from keras import backend as K
from keras.models import model_from_json
from keras.utils import np_utils
import numpy as np
import h5py
import pickle
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

def get_activations(model, layer, X_batch):
	"""
	Purpose -> Obtains outputs from any layer in Keras
	Input   -> Trained model, layer from which output needs to be extracted & files to be given as input
	Output  -> Features from that layer 
	"""
	#Referred from:- TODO: Enter the forum link from where I got this
	get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
	activations = get_activations([X_batch,0])
	return activations

experiment_details = 'lstm128_subword'
Masterdir = '/media/ameya/Research/Sub-word-LSTM/'
Datadir = 'Data/'
Modeldir = 'Models/'
Featuredir = 'Features/'

batch_size = 128
numclasses = 3

mapping_char2num = {' ': 0, 'a': 1, 'c': 2, 'b': 3, 'e': 4, 'd': 5, 'g': 6, 'f': 7, 'i': 8, 'h': 9, 'k': 10, 'j': 11, 'm': 12, 'l': 13, 'o': 14, 'n': 15, 'q': 16, 'p': 17, 's': 18, 'r': 19, 'u': 20, 't': 21, 'w': 22, 'v': 23, 'y': 24, 'x': 25, 'z': 26}
mapping_num2char = {0: ' ', 1: 'a', 2: 'c', 3: 'b', 4: 'e', 5: 'd', 6: 'g', 7: 'f', 8: 'i', 9: 'h', 10: 'k', 11: 'j', 12: 'm', 13: 'l', 14: 'o', 15: 'n', 16: 'q', 17: 'p', 18: 's', 19: 'r', 20: 'u', 21: 't', 22: 'w', 23: 'v', 24: 'y', 25: 'x', 26: 'z'}

mapping_char2num = {' ': 0, 'a': 1, 'c': 2, 'b': 3, 'e': 4, 'd': 5, 'g': 6, 'f': 7, 'i': 8, 'h': 9, 'k': 10, 'j': 11, 'm': 12, 'l': 13, 'o': 14, 'n': 15, 'q': 16, 'p': 17, 's': 18, 'r': 19, 'u': 20, 't': 21, 'w': 22, 'v': 23, 'y': 24, 'x': 25, 'z': 26}
mapping_num2char = {0: ' ', 1: 'a', 2: 'c', 3: 'b', 4: 'e', 5: 'd', 6: 'g', 7: 'f', 8: 'i', 9: 'h', 10: 'k', 11: 'j', 12: 'm', 13: 'l', 14: 'o', 15: 'n', 16: 'q', 17: 'p', 18: 's', 19: 'r', 20: 'u', 21: 't', 22: 'w', 23: 'v', 24: 'y', 25: 'x', 26: 'z'}


h5f = h5py.File(Masterdir+Datadir+'Xtest_'+experiment_details+'.h5','r')
X_test = h5f['dataset'][:]
h5f.close()
print(X_test.shape)

inp = open(Masterdir+Datadir+'Ytest_'+experiment_details+'.pkl', 'rb')
y_test=pickle.load(inp)
inp.close()
y_test=np.asarray(y_test).flatten()
y_test = np_utils.to_categorical(y_test, numclasses) 
print(y_test.shape)

f = open(Masterdir+Modeldir+'LSTM_'+experiment_details+'_architecture.json','r+')
json_string = f.read()
f.close()
model = model_from_json(json_string)

model.load_weights(Masterdir+Modeldir+'LSTM_'+experiment_details+'_weights.h5')
model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])

activations = get_activations(model, 1, X_test)
features_test = np.asarray(activations)

scipy.io.savemat(Masterdir+Featuredir+'Visualizations.mat', mdict={'exon': features_test})	

f=open(Masterdir+Featuredir+'Write_Xtest.txt','w')
for line in X_test:
	#print(line)
	s=''
	for num in line:
		s=s+mapping_num2char[int(num)]
	#print(s)
	f.write(s+'\n')
f.close()