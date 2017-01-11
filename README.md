# Sub-word-LSTM

This repository contains all codes for reproducing the results for our COLING paper. We hope that releasing the source code and pre-trained models are beneficial to the community, and recieve constructive feedback on the same.

Details of the networks are given in [our paper](https://arxiv.org/abs/1611.00472 "Research paper"), and if you think this is helpful in your research, cite our paper:
```
Towards Sub-Word Level Compositions for Sentiment Analysis of Hi-En Code Mixed Text, COLING 2016.

Ameya Prabhu*, Aditya Joshi*, Manish Shrivastava and Vasudeva Verma
(* indicating equal contribution)


@article{prabhu2016subword,
  title={Towards Sub-Word Level Compositions for Sentiment Analysis of Hindi-English Code Mixed Text},
  author={Prabhu, Ameya, Joshi, Aditya, Shrivastava, Manish and Verma, Vasudeva},
  journal={arXiv preprint arXiv:1611.00472},
  year={2016}
}
```

## Concept
TBA

## IIITH Codemixed Sentiment Dataset

The dataset is available in the data/ directory. The original dataset is IIITH_Codemixed.txt and also the preprocessed ones are available alongside organized as X_train, X_test, Y_train, Y_test (The dictionaries to map the numbers to characters are given in the training script.)

## Using the code in this repository

Step 1: Clone the repository (And if you like it, star it too!)
Step 2: Goto Code/ and open the file you want to run.
Step 3: Enter the folder in which you have placed this repository into Masterdir variable.
And you're good to go! Sample scripts are given in the repository (my original experiments). 
You can just go run them directly as a starting point, and then proceed to tinker around!

## Training you own model

Follow the instructions as given above and after that to run the script:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32, allow_gc=False python char_rnn_train.py
```
Using a GPU is recommended (It'll reduce the training time significantly).

## Testing a trained model

Follow the instructions as given above, place your model in the Pretrained_models/ directory naming it as the model given alongside the code and after that to run the script:
```
python char_rnn_test.py
```
The scripts are easily modifiable (Atleast I think they are!). Feel free to play around with the codes, models and visualizations!

## Visualizing the feature maps

Follow the instructions as given above, place your model in the Pretrained_models/ directory naming it as the model given alongside the code and after that to run the script:
```
python visualize.py
```
It will save the convolution outputs in a .mat file. You can go ahead, open matlab and to view the output type:
```
*load the mat file and select the image you want to display in a variable im*
imshow(im');
```
