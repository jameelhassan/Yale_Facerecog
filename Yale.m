clear all; close all;
clc;

%%
data = load("Yale_32x32.mat", 'fea', 'gnd');
indata = data.fea;
out = data.gnd;
num_faces = max(out);

all_combos = eye(num_faces);
Y = all_combos(out,:);
Y = Y';
X = indata';

net = feedforwardnet([60 20]);

%% Neural Net with output vals ranging 1-15
% net = configure(net,X,out');
% view(net);
% load('nn_1thru15.mat');
% stem(round(nn(X))); 
% hold on; 
% stem(out');

%% Neural Net with output as a 1x15 sparse vector
net = configure(net,X,Y);
net.layers{3}.transferFcn = 'softmax';
net.performFcn = 'crossentropy'
net.trainFcn = 'trainscg'
view(net)

%Evaluate trained parameters
load('NN_sparsevec.mat');
load('trainstruct.mat');

testX = X(:,tr.testInd);
testY = Y(:,tr.testInd);
testeval = nn(testX);
plotconfusion(testY,testeval)
