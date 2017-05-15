import numpy as np
import mxnet as mx

import plot_bench2
import pickle
from datetime import datetime, date
import sys
import logging
import os

from sklearn.utils import shuffle
import matplotlib.pyplot as plt


#define log loss
def logloss(label, pred):
    #print label.shape
    #print pred.shape
    value=-np.mean(label*np.log(pred+1.0e-6)+(1.0-label)*np.log(1.0-pred+1.0e-6))
    return value

def augmentdata(datain, dataout, nrot=3):
    # make copy
    #copydata = np.swapaxes(datain, 0, 2)
    copydata = np.swapaxes(datain, 0, 1)
    copydata = np.fliplr(copydata)
    #result = np.swapaxes(copydata, 0, 2)
    result = np.swapaxes(copydata, 0, 1)
    copyofdataout = np.array(dataout)    
    return result, copyofdataout
    


logging.basicConfig(
            format="%(message)s",
            level=logging.DEBUG,
            stream=sys.stdout)

trainnb=0.9 # Fraction used for training

print 'Pickling out'
#train_p=np.load('H_top_feature_aligned_new_800_30_large.npy').astype(np.float32) #positive samples (top)
train_p=np.load('H_top_feature_aligned_new_800_30_clean01_test.npy').astype(np.float32) #positive samples (top)
#train_n=np.load('H_dijet_feature_aligned_new_800_30_large.npy').astype(np.float32) #negative samples (qcd)
train_n=np.load('H_dijet_feature_aligned_new_800_30_clean01_test.npy').astype(np.float32) #negative samples (qcd)

train_data=np.vstack((train_p,train_n))
train_out=np.array([1]*len(train_p)+[0]*len(train_n))

augdata, augout = augmentdata(train_data, train_out)

print "train_out before", train_out
print "len= ", len(train_out)

train_data=np.vstack((train_data, augdata))
train_out=np.append(train_out, augout)

print "train_out after", train_out

numbertr=len(train_out)

print "numbertr= ", numbertr

#Shuffling
order=shuffle(range(numbertr),random_state=100)
train_out=train_out[order]
train_data=train_data[order,0::]

print "train_out shuflling", train_out


# Energy normalization
for j in range(len(train_data)):
    #print j, " " , train_data[j,0::]
    train_data[j,0::]=train_data[j,0::]/np.sum(train_data[j,0::])*10.0

#Splitting between training set and cross-validation set
valid_data=train_data[int(trainnb*numbertr):numbertr,0::]
valid_data_out=train_out[int(trainnb*numbertr):numbertr]

train_data_out=train_out[0:int(trainnb*numbertr)]
train_data=train_data[0:int(trainnb*numbertr),0::]

train_data=train_data.reshape(len(train_data), 1, 30, 30)
valid_data=valid_data.reshape(len(valid_data), 1, 30, 30)


#Initialize weights of the samples
pos_s=train_data_out==1
pos_b=train_data_out==0
train_weights=np.ones(len(train_data_out))
train_balance_weights=train_weights*(pos_s.astype(float)*(1.0/np.sum(train_weights[pos_s]))\
                                     +pos_b.astype(float)*(1.0/np.sum(train_weights[pos_b])))
train_balance_weights=train_balance_weights/np.sum(train_balance_weights)*len(train_balance_weights)

# mxnet
# reasonable choices 30,30 relu
data = mx.symbol.Variable('data')

#good 
"""
conv1 = mx.symbol.Convolution(data=data, name='conv1', kernel=(5,5), stride=(2,2), num_filter=20)
cnnact1 = mx.symbol.Activation(data = conv1, name='cnnact1', act_type="relu")
pool1 = mx.symbol.Pooling(data=cnnact1, name='pool1', kernel=(2,2),stride=(1,1),pool_type='max')
conv2 = mx.symbol.Convolution(data=pool1, name='conv2', kernel=(3,3), num_filter=20)
cnnact2 = mx.symbol.Activation(data = conv2, name='cnnact2', act_type="relu")
#flatten = mx.symbol.Flatten(pool1)
flatten = mx.symbol.Flatten(cnnact2)
cnnfc1 = mx.symbol.FullyConnected(data = flatten, name = 'cnnfc1', num_hidden = 32)
fcact1 = mx.symbol.Activation(data = cnnfc1, name='fcact1', act_type="sigmoid")
#cnnfc2 = mx.symbol.FullyConnected(data = fcact1, name = 'cnnfc2', num_hidden = 16)
#fcact2 = mx.symbol.Activation(data = cnnfc2, name='fcact2', act_type="sigmoid")
cnnfc3 = mx.symbol.FullyConnected(data = fcact1, name = 'cnnfc3', num_hidden = 2)
fcact3 = mx.symbol.Activation(data = cnnfc3, name='fcact3', act_type="sigmoid")
mlp2 = mx.symbol.SoftmaxOutput(data = fcact3, name = 'softmax')
"""
conv1 = mx.symbol.Convolution(data=data, name='conv1', kernel=(7,7), stride=(1,1), num_filter=20)
cnnact1 = mx.symbol.Activation(data = conv1, name='cnnact1', act_type="relu")
pool1 = mx.symbol.Pooling(data=cnnact1, name='pool1', kernel=(2,2), stride=(2,2),pool_type='max')
conv2 = mx.symbol.Convolution(data=pool1, name='conv2', kernel=(5,5), num_filter=30)
cnnact2 = mx.symbol.Activation(data = conv2, name='cnnact2', act_type="relu")
flatten = mx.symbol.Flatten(cnnact2)
cnnfc1 = mx.symbol.FullyConnected(data = flatten, name = 'cnnfc1', num_hidden = 32)
fcact1 = mx.symbol.Activation(data = cnnfc1, name='fcact1', act_type="relu")
#cnnfc2 = mx.symbol.FullyConnected(data = fcact1, name = 'cnnfc2', num_hidden = 16)
#fcact2 = mx.symbol.Activation(data = cnnfc2, name='fcact2', act_type="sigmoid")
cnnfc3 = mx.symbol.FullyConnected(data = fcact1, name = 'cnnfc3', num_hidden = 2)
fcact3 = mx.symbol.Activation(data = cnnfc3, name='fcact3', act_type="relu")
mlp2 = mx.symbol.SoftmaxOutput(data = fcact3, name = 'softmax')

#create group of outputs for

epochs=[]
epochs2=[]
validationperf=[]
trainingperf=[]

def evalbatchendcallback(batchendparams):
    epochs.append(batchendparams.epoch)
    value = batchendparams.eval_metric.get()[1]
    validationperf.append(value)
    
def batchendcallback(params):
    epochs2.append(params.epoch)
    value = params.eval_metric.get()[1]
    trainingperf.append(value)
        
def plotbatchperformance():
    plt.figure()
    plt.plot(epochs, validationperf, color='red')
    plt.plot(epochs2, trainingperf, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Perfrmance Metric')
    plt.title('Neural network training')
    plt.show()

# plot convolution filters
def plotconvfilters(filters):
    plt.figure()
    numfilters=len(filters)
    nrows = numfilters/5+1
    for ifilter in xrange(numfilters):
        plt.subplot(nrows, 5, ifilter+1)
        ai=plt.imshow(filters[ifilter,0], cmap='RdYlGn', vmin=-1.0, vmax=1.0, interpolation='none')
        plt.colorbar()
    plt.show()
logging.basicConfig(level=logging.DEBUG)

def norm_stat(d):
    return mx.nd.norm(d)/np.sqrt(d.size)
mon = mx.mon.Monitor(100, norm_stat)

if os.path.exists('cnn-mxnet.pkl'):
    model=pickle.load(open('cnn-mxnet.pkl', 'rb'))
else:
    model = mx.model.FeedForward(
        #ctx = mx.gpu(),
        initializer = mx.init.Xavier(),
        optimizer = mx.optimizer.Adam(),
        symbol = mlp2, 
        num_epoch = 10,
        numpy_batch_size=10000,
        learning_rate = 0.5, 
        momentum = 0.95, wd = 0.0001
        )
    print "model"
    try:
        model.fit(X=train_data, y=train_data_out, eval_data=(valid_data,valid_data_out), monitor=mon, 
                  #eval_metric=mx.metric.np(logloss),
                  eval_metric='ce',
                  eval_batch_end_callback=evalbatchendcallback,
                  batch_end_callback = batchendcallback)
    except KeyboardInterrupt:
        pass
    pickle.dump(model, open('cnn-mxnet.pkl', 'wb'))
    print "Performance plot"
    plotbatchperformance()

output_valid = model.predict(valid_data)[0::,1]

valid_data_out=np.ravel(valid_data_out)
output_valid = np.ravel(output_valid)
print "Plot"
#plotbatchperformance()

convweights = model.arg_params.get('conv1_weight').asnumpy()
plotconvfilters(convweights)

plot_bench2.plot(output_valid,valid_data_out)

