from keras.models import model_from_json
from keras.utils import np_utils
import numpy as np
import h5py
import pickle
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

Masterdir = '/media/ameya/Research/Sub_word_LSTM/'
Datadir = 'Data/'
Modeldir = 'Models/'
Featuredir = 'Features/'
inputdatasetfilename = 'IIITH_Codemixed.txt'
experiment_details = 'char_rnn_emnlp_lstm128'
filename = 'match.txt'
batch_size = 128
numclasses = 3

def accuracy(original, predicted):
	print("F1 score is: " + str(f1_score(original, predicted, average='macro')))
	scores = confusion_matrix(original, predicted)
	print scores
	print np.trace(scores)/float(np.sum(scores))

h5f = h5py.File(masterdir+charrnndir+'Data/Xtest_'+experiment_details+'.h5','r')
X_test = h5f['dataset'][:]
h5f.close()
print(X_test.shape)

inp = open(masterdir+charrnndir+'Data/Ytest_'+experiment_details+'.pkl', 'rb')
y_test=pickle.load(inp)
inp.close()
y_test=np.asarray(y_test).flatten()
y_test2 = np_utils.to_categorical(y_test, numclasses) 
print(y_test.shape)
f = open(masterdir+charrnndir+'Models/'+'LSTM_'+experiment_details+'_architecture.json','r+')
json_string = f.read()
f.close()
model = model_from_json(json_string)

model.load_weights(masterdir+charrnndir+'Models/LSTM_'+experiment_details+'_weights.h5')
model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
	

score, acc = model.evaluate(X_test, y_test2, batch_size=batch_size)

y_pred = model.predict_classes(X_test, batch_size=batch_size)
#print(y_pred)
accuracy(y_test,y_pred)

print('Accuracy is: '+str(acc))