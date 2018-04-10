%% load AlexNet
cnnURL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-caffe-alex.mat';
cnnMatFile = 'imagenet-caffe-alex.mat';
% Load MatConvNet network into a SeriesNetwork
convnet = helperImportMatConvNet(cnnMatFile);
net = convnet.Layers
%% setting weights
%% use fully connected
featureLayer = 'fc7';
%% load dataset
imds = imageDatastore('C:\Z_Data\QuickDrawData\Tommy', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%% resize the image to 227 *227*3
inputSize = [227 227];
imds.ReadFcn = @(loc)imresize(imread(loc),inputSize);
imshow(preview(imds))
%%
split for trainingset and testingset
T = countEachLabel(imds);
[imdsTrain,imdsTest] = splitEachLabel(imds,0.75)
inputSize=net(1).InputSize
%% train
trainsferLayer = net(1:end-3);
layers = [trainsferLayer;
    fullyConnectedLayer(4,'WeightLearnRateFactor',50,'BiasLearnRateFactor',50);%??????????????????5
    softmaxLayer();
    classificationLayer()];

options = trainingOptions('sgdm',...
    'Maxepochs',4,...
    'InitialLearnRate',0.0001);

network = trainNetwork(imdsTrain,layers,options);

%% predict
predictLabels = classify(network,imdsTest);
%%
in=[227 227]
read=imread('cell_phone1.png');
im = imresize(read,in);
imshow(im)
[YPred,scores] = classify(network,im);
