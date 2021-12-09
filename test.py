import random
import numpy as np
import math
import torch
import torch.nn as nn
from collections import OrderedDict
from itertools import product as product
from caffe2pth.prototxt import *
from caffe2pth.detection import Detection, MultiBoxLoss
import caffe_pb2
from caffe2pth.caffenet import *
from torchnet import FeatNet, MaskNet
import cv2

def parse_caffemodel(caffemodel):
    model = caffe_pb2.NetParameter()
    with open(caffemodel, 'rb') as fp:
        model.ParseFromString(fp.read())
    return model

def load_weights_featnet(protofile, caffemodel,net):
    model = caffemodel
    layers = model.layer
    lmap = {}
    for l in layers:
        lmap[l.name] = l
    net_info = parse_prototxt(protofile)
    layers = net_info['layers']
    def find_caffe_weight(index,conv_layer):
        caffe_weight = np.array(lmap[layers[index]['name']].blobs[0].data)    # i is the index of the specific layer we want to operate
        caffe_weight = torch.from_numpy(caffe_weight).view_as(conv_layer.weight)
        return caffe_weight
    net.conv1.weight.data.copy_(find_caffe_weight(index=1,conv_layer=net.conv1))
    net.conv2.weight.data.copy_(find_caffe_weight(index=4,conv_layer=net.conv2))
    net.conv3.weight.data.copy_(find_caffe_weight(index=8,conv_layer=net.conv3))
    net.conv4.weight.data.copy_(find_caffe_weight(index=12,conv_layer=net.conv4))
    net.Upsample2.weight.data.copy_(find_caffe_weight(index=6,conv_layer=net.Upsample2))
    net.Upsample3.weight.data.copy_(find_caffe_weight(index=10,conv_layer=net.Upsample3))
        
def load_weights_masknet(protofile, caffemodel,net):
    model = caffemodel
    layers = model.layer
    lmap = {}
    for l in layers:
        lmap[l.name] = l
    net_info = parse_prototxt(protofile)
    layers = net_info['layers']
    def find_caffe_weight(index,conv_layer):
        caffe_weight = np.array(lmap[layers[index]['name']].blobs[0].data)    # i is the index of the specific layer we want to operate
        caffe_weight = torch.from_numpy(caffe_weight).view_as(conv_layer.weight)
        return caffe_weight
    def find_caffe_bias(index):
        caffe_bias = torch.from_numpy(np.array(lmap[layers[index]['name']].blobs[1].data))
        return caffe_bias
    net.conv1.weight.data.copy_(find_caffe_weight(index=13,conv_layer=net.conv1))
    net.conv1.bias.data.copy_(find_caffe_bias(index=13))
    net.conv2.weight.data.copy_(find_caffe_weight(index=16,conv_layer=net.conv2))
    net.conv2.bias.data.copy_(find_caffe_bias(index=16))
    net.conv3.weight.data.copy_(find_caffe_weight(index=20,conv_layer=net.conv3))
    net.conv3.bias.data.copy_(find_caffe_bias(index=20))
    net.conv4.weight.data.copy_(find_caffe_weight(index=24,conv_layer=net.conv4))
    net.conv4.bias.data.copy_(find_caffe_bias(index=24))
    net.conv2s.weight.data.copy_(find_caffe_weight(index=18,conv_layer=net.conv2s))
    net.conv2s.bias.data.copy_(find_caffe_bias(index=18))
    net.conv3s.weight.data.copy_(find_caffe_weight(index=22,conv_layer=net.conv3s))
    net.conv3s.bias.data.copy_(find_caffe_bias(index=22))
    net.conv4s.weight.data.copy_(find_caffe_weight(index=26,conv_layer=net.conv4s))
    net.conv4s.bias.data.copy_(find_caffe_bias(index=26))
    net.Upsample1.weight.data.copy_(find_caffe_weight(index=27,conv_layer=net.Upsample1))
    net.Upsample2.weight.data.copy_(find_caffe_weight(index=29,conv_layer=net.Upsample2))
    net.Upsample3.weight.data.copy_(find_caffe_weight(index=31,conv_layer=net.Upsample3))


if __name__ == '__main__':
    protofile = 'a.prototxt'
    caffemodel = parse_caffemodel('casia.caffemodel')
    featnet = FeatNet()
    masknet = MaskNet()
    load_weights_featnet(protofile,caffemodel,featnet)
    load_weights_masknet(protofile,caffemodel,masknet)
    torch.save(featnet.state_dict(),'featnet.pth')
    torch.save(masknet.state_dict(),'masknet.pth')

    featnet.load_state_dict(torch.load('featnet.pth'))
    masknet.load_state_dict(torch.load('masknet.pth'))

    img = cv2.imread('1.bmp',cv2.IMREAD_GRAYSCALE)
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    img = img/255 - 0.25
    iriscode = featnet(img)
    iriscode = iriscode.detach().numpy().squeeze(0).squeeze(0)
    irismask = masknet(img)
    irismask = (irismask.detach().squeeze(0))[1,:,].squeeze(0).numpy()    
    cv2.imshow('iriscode.jpg',iriscode)
    cv2.imshow('irismask.jpg',irismask)
    cv2.waitKey()