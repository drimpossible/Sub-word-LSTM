from keras.models import model_from_json
from keras.utils import np_utils
import numpy as np
import h5py
import pickle
from keras import backend as K
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from MyNormalizer import token
from keras.preprocessing import sequence
from skimage import io
import matplotlib.pyplot as plt

Masterdir = '/media/ameya/Research/Sub-word-LSTM/'
Datadir = 'Data/'
Modeldir = 'Pretrained_models/'
Featuredir = 'Features/'
inputdatasetfilename = 'IIITH_Codemixed.txt'
experiment_details = 'lstm128_subword'
filename = 'match.txt'
batch_size = 128
numclasses = 3

mapping_char2num = {' ': 0, 'a': 1, 'c': 2, 'b': 3, 'e': 4, 'd': 5, 'g': 6, 'f': 7, 'i': 8, 'h': 9, 'k': 10, 'j': 11, 'm': 12, 'l': 13, 'o': 14, 'n': 15, 'q': 16, 'p': 17, 's': 18, 'r': 19, 'u': 20, 't': 21, 'w': 22, 'v': 23, 'y': 24, 'x': 25, 'z': 26}
mapping_num2char = {0: ' ', 1: 'a', 2: 'c', 3: 'b', 4: 'e', 5: 'd', 6: 'g', 7: 'f', 8: 'i', 9: 'h', 10: 'k', 11: 'j', 12: 'm', 13: 'l', 14: 'o', 15: 'n', 16: 'q', 17: 'p', 18: 's', 19: 'r', 20: 'u', 21: 't', 22: 'w', 23: 'v', 24: 'y', 25: 'x', 26: 'z'}

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
#h5f = h5py.File(Masterdir+Datadir+'Xtest_'+experiment_details+'.h5','r')
#X_test = h5f['dataset'][:]
#h5f.close()
#print(X_test.shape)
#
#inp = open(Masterdir+Datadir+'Ytest_'+experiment_details+'.pkl', 'rb')
#y_test=pickle.load(inp)
#inp.close()
#y_test=np.asarray(y_test).flatten()
#y_test2 = np_utils.to_categorical(y_test, numclasses) 
#print(y_test.shape)
f = open(Masterdir+Modeldir+'LSTM_'+experiment_details+'_architecture.json','r+')
json_string = f.read()
f.close()
model = model_from_json(json_string)

model.load_weights(Masterdir+Modeldir+'LSTM_'+experiment_details+'_weights.h5')
model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
	
while(1):
	inp_sent = raw_input('Enter a sentence. Press \'Q\' to exit.\n')
	if inp_sent == "Q":
		break
	inp_sent = token(inp_sent)
	X_test = []
	temp = []
	for words in inp_sent:
		for char in words:
			temp.append(mapping_char2num[char])
		temp.append(mapping_char2num[' '])
	X_test.append(temp)
	X_test = np.asarray(X_test)
	print(X_test.shape)

	X_test = sequence.pad_sequences(X_test[:], maxlen=200)
	print(X_test.shape)
	#score, acc = model.evaluate(X_test, y_test2, batch_size=batch_size)

	y_pred = model.predict_classes(X_test, batch_size=batch_size)

	activations = get_activations(model, 1, X_test)
	features_test = np.squeeze(np.asarray(activations))
	print(features_test.shape)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for line in X_test:
		s=''
		for num in line:
			s=s+mapping_num2char[int(num)]
	ax.text(0,-1,s,fontsize=5.52,family='monospace')
	ax.imshow(features_test.T)
	plt.show()
	if y_pred == 1:
		print('Neutral is the prediction!')
	if y_pred == 0:
		print('Negative is the prediction!')
	if y_pred == 2:
		print('Positive is the prediction!')