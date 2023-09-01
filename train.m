clc
clear all
close all
warning off
g=alexnet;%pretrained network deep learning architecture 
layers=g.Layers;%layer extraction

layers(23)=fullyConnectedLayer(2);
layers(25)=classificationLayer;
imds=imageDatastore('database','IncludeSubfolders',true, 'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);

%augimdsTrain = augmentedImageDatastore([224 224],imdsTrain,'DataAugmentation',augmenter);
%augimdsValidation = augmentedImageDatastore([224 224],imdsValidation,'DataAugmentation',augmenter);
augimdsTrain = augmentedImageDatastore([227 227],imdsTrain);
augimdsValidation = augmentedImageDatastore([227 227],imdsValidation);
opts=trainingOptions('sgdm','InitialLearnRate',0.0001,'MaxEpochs',5,'MiniBatchSize',128);
myNet2 = trainNetwork(augimdsTrain,layers,opts);
save myNet2; 

