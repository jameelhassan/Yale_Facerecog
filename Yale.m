clear all; close all;
clc;

data = load("Yale_32x32.mat", 'fea', 'gnd');
indata = data.fea;
out = data.gnd;
num_faces = max(out);

all_combos = eye(num_faces);
Y = all_combos(out,:);
Y = Y';
X = indata';

net = feedforwardnet([10,5]);
net = configure(net,X,out');
view(net);
