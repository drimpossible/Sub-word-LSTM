import numpy as np
import h5py
import pickle
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.preprocessing import sequence
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Convolution1D, MaxPooling1D

from keras.utils import np_utils
from MyNormalizer import token

################# GLOBAL VARIABLES #####################
#Filenames
#TODO: Add to coding conventions that directories are to always end with '/'
Masterdir = '/media/ameya/Research/Sub-word-LSTM/'
Datadir = 'Data/'
Modeldir = 'Models/'
Featuredir = 'Features/'
inputdatasetfilename = 'IIITH_Codemixed.txt'
exp_details = 'new_experiment'

#Data I/O formatting
SEPERATOR = '\t'
DATA_COLUMN = 1
LABEL_COLUMN = 3
LABELS = ['0','1','2'] # 0 -> Negative, 1-> Neutral, 2-> Positive
mapping_char2num = {}
mapping_num2char = {}
MAXLEN = 200

#LSTM Model Parameters
#Embedding
MAX_FEATURES = 0
embedding_size = 128
# Convolution
filter_length = 3
nb_filter = 128
pool_length = 3
# LSTM
lstm_output_size = 128
# Training
batch_size = 128
number_of_epochs = 50
numclasses = 3
test_size = 0.2
########################################################

def parse(Masterdir,filename,seperator,datacol,labelcol,labels):
	"""
	Purpose -> Data I/O
	Input   -> Data file containing sentences and labels along with the global variables
	Output  -> Sentences cleaned up in list of lists format along with the labels as a numpy array
	"""
	#Reads the files and splits data into individual lines
	f=open(Masterdir+Datadir+filename,'r')
	lines = f.read().lower()
	lines = lines.lower().split('\n')[:-1]

	X_train = []
	Y_train = []
	
	#Processes individual lines
	for line in lines:
		# Seperator for the current dataset. Currently '\t'. 
		line = line.split(seperator)
		#Token is the function which implements basic preprocessing as mentioned in our paper
		tokenized_lines = token(line[datacol])
		
		#Creates character lists
		char_list = []
		for words in tokenized_lines:
			for char in words:
				char_list.append(char)
			char_list.append(' ')
		#print(char_list) - Debugs the character list created
		X_train.append(char_list)
		
		#Appends labels
		if line[labelcol] == labels[0]:
			Y_train.append(0)
		if line[labelcol] == labels[1]:
			Y_train.append(1)
		if line[labelcol] == labels[2]:
			Y_train.append(2)
	
	#Converts Y_train to a numpy array	
	Y_train = np.asarray(Y_train)
	assert(len(X_train) == Y_train.shape[0])

	return [X_train,Y_train]

def convert_char2num(mapping_n2c,mapping_c2n,trainwords,maxlen):
	"""
	Purpose -> Convert characters to integers, a unique value for every character
	Input   -> Training data (In list of lists format) along with global variables
	Output  -> Converted training data along with global variables
	"""
	allchars = []
	errors = 0

	#Creates a list of all characters present in the dataset
	for line in trainwords:
		try:
			allchars = set(allchars+line)
			allchars = list(allchars)
		except:
			errors += 1

	#print(errors) #Debugging
	#print(allchars) #Debugging 

	#Creates character dictionaries for the characters
	charno = 0
	for char in allchars:
		mapping_char2num[char] = charno
		mapping_num2char[charno] = char
		charno += 1

	assert(len(allchars)==charno) #Checks

	#Converts the data from characters to numbers using dictionaries 
	X_train = []
	for line in trainwords:
		char_list=[]
		for letter in line:
			char_list.append(mapping_char2num[letter])
		#print(no) -- Debugs the number mappings
		X_train.append(char_list)
	print(mapping_char2num)
	print(mapping_num2char)
	#Pads the X_train to get a uniform vector
	#TODO: Automate the selection instead of manual input
	X_train = sequence.pad_sequences(X_train[:], maxlen=maxlen)
	return [X_train,mapping_num2char,mapping_char2num,charno]

def RNN(X_train,y_train,args):
	"""
	Purpose -> Define and train the proposed LSTM network
	Input   -> Data, Labels and model hyperparameters
	Output  -> Trained LSTM network
	"""
	#Sets the model hyperparameters
	#Embedding hyperparameters
	max_features = args[0]
	maxlen = args[1]
	embedding_size = args[2]
	# Convolution hyperparameters
	filter_length = args[3]
	nb_filter = args[4]
	pool_length = args[5]
	# LSTM hyperparameters
	lstm_output_size = args[6]
	# Training hyperparameters
	batch_size = args[7]
	nb_epoch = args[8]
	numclasses = args[9]
	test_size = args[10] 

	#Format conversion for y_train for compatibility with Keras
	y_train = np_utils.to_categorical(y_train, numclasses) 
	#Train & Validation data splitting
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=test_size, random_state=42)
	
	#Build the sequential model
	# Model Architecture is:
	# Input -> Embedding -> Conv1D+Maxpool1D -> LSTM -> LSTM -> FC-1 -> Softmaxloss
	print('Build model...')
	model = Sequential()
	model.add(Embedding(max_features, embedding_size, input_length=maxlen))
	model.add(Convolution1D(nb_filter=nb_filter,
							filter_length=filter_length,
							border_mode='valid',
							activation='relu',
							subsample_length=1))
	model.add(MaxPooling1D(pool_length=pool_length))
	model.add(LSTM(lstm_output_size, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
	model.add(LSTM(lstm_output_size, dropout_W=0.2, dropout_U=0.2, return_sequences=False))
	model.add(Dense(numclasses))
	model.add(Activation('softmax'))

	# Optimizer is Adamax along with categorical crossentropy loss
	model.compile(loss='categorical_crossentropy',
			  	optimizer='adamax',
			  	metrics=['accuracy'])
	

	print('Train...')
	#Trains model for 50 epochs with shuffling after every epoch for training data and validates on validation data
	model.fit(X_train, y_train, 
			  batch_size=batch_size, 
			  shuffle=True, 
			  nb_epoch=nb_epoch,
			  validation_data=(X_valid, y_valid))
	return model

def save_model(Masterdir,filename,model):
	"""
	Purpose -> Saves Keras model files to the given directory
	Input   -> Directory and experiment details to be saved and trained model file
	Output  -> Nil
	"""
	#Referred from:- http://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
	model.save_weights(Masterdir+charrnndir+'Models/LSTM_'+filename+'_weights.h5')
	json_string = model.to_json()
	f = open(Masterdir+charrnndir+'Models/'+'LSTM_'+filename+'_architecture.json','w')
	f.write(json_string)
	f.close()

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

def evaluate_model(X_test,y_test,model,batch_size,numclasses):
	"""
	Purpose -> Evaluate any model on the testing data
	Input   -> Testing data and labels, trained model and global variables
	Output  -> Nil
	"""
	#Convert y_test to one-hot encoding
	y_test = np_utils.to_categorical(y_test, numclasses)
	#Evaluate the accuracies
	score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
	print('Test score:', score)
	print('Test accuracy:', acc)

def save_data(Masterdir,filename,X_train,X_test,y_train,y_test,features_train,features_test):
	"""
	Purpose -> Saves train, test data along with labels and features in the respective directories in the folder
	Input   -> Train and test data, labels and features along with the directory and experiment details to be mentioned
	Output  -> Nil
	"""
	h5f = h5py.File(Masterdir+Datadir+'Xtrain_'+filename+'.h5', 'w')
	h5f.create_dataset('dataset', data=X_train)
	h5f.close()

	h5f = h5py.File(Masterdir+Datadir+'Xtest_'+filename+'.h5', 'w')
	h5f.create_dataset('dataset', data=X_test)
	h5f.close()

	output = open(Masterdir+Datadir+'Ytrain_'+filename+'.pkl', 'wb')
	pickle.dump([y_train], output)
	output.close()

	output = open(Masterdir+Datadir+'Ytest_'+filename+'.pkl', 'wb')
	pickle.dump([y_test], output)
	output.close()

	h5f = h5py.File(Masterdir+Featuredir+'features_train_'+filename+'.h5', 'w')
	h5f.create_dataset('dataset', data=features_train)
	h5f.close()

	h5f = h5py.File(Masterdir+Featuredir+'features_test_'+filename+'.h5', 'w')
	h5f.create_dataset('dataset', data=features_test)
	h5f.close()

if __name__ == '__main__':
	"""
	Master function
	"""
	print('Starting RNN Engine...\nModel: Char-level LSTM.\nParsing data files...')
	out = parse(Masterdir,inputdatasetfilename,SEPERATOR,DATA_COLUMN,LABEL_COLUMN,LABELS)
	X_train = out[0]
	y_train = out[1]
	print('Parsing complete!')

	print('Creating character dictionaries and format conversion in progess...')
	out = convert_char2num(mapping_num2char,mapping_char2num,X_train,MAXLEN)
	mapping_num2char = out[1]
	mapping_char2num = out[2]
	MAX_FEATURES = out[3]
	X_train = np.asarray(out[0])
	y_train = np.asarray(y_train).flatten()
	print('Complete!')
	
	print('Splitting data into train and test...')
	X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
	print('X_train shape:', X_train.shape)
	print('X_test shape:', X_test.shape)
	
	print('Creating LSTM Network...')
	model = RNN(deepcopy(X_train),deepcopy(y_train),[MAX_FEATURES, MAXLEN, embedding_size,\
			     filter_length, nb_filter, pool_length, lstm_output_size, batch_size, \
			     number_of_epochs, numclasses, test_size])

	print('Evaluating model...')
	evaluate_model(X_test,deepcopy(y_test),model,batch_size,numclasses)
	
	print('Feature extraction pipeline running...')
	activations = get_activations(model, 4, X_train)
	features_train = np.asarray(activations)
	activations = get_activations(model, 4, X_test)
	features_test = np.asarray(activations)
	print('Features extracted!')
	
	print('Saving experiment...')
	save_model(Masterdir,exp_details,model)
	save_data(Masterdir,exp_details,X_train,X_test,y_train,y_test,features_train,features_test)
	print('Saved! Experiment finished!')