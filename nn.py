import pandas as pd
import numpy as np
import keras

import matplotlib.pyplot as plt

from keras.layers import Dense
from keras.engine.input_layer import Input

import sklearn.metrics

from keras.utils import plot_model

data = pd.read_csv('data.txt',delimiter='\t')

data=data.rename(columns={'Rejects':'Output'})

#Shuffle inputs
data=data.sample(frac=1).reset_index()



def main():
    print (data)

    #one-hot encoding
    data['Process_0']=data['Process']==0
    data['Process_1']=data['Process']==1


    mean=data['Output'].min()
    mean=0
    data['Output'] -= mean

    std = data['Output'].max()
    std=1
    data['Output'] /= std


    test_mode=True


    if test_mode:

        #Data is very limited so I can do this

        optimizers=['adam','adagrad','RMS']
        learning_rate=[.001,.005,.01,.02]
        losses=['mean_absolute_error','mean_squared_error','mean_squared_logarithmic_error','hinge']
        activations=['sigmoid','tan_h','relu','leaky_relu']

        batch_size = [10,50,100,200]
        epoch_ct = 20

        min_loss = 1e6
        min_hyperparams= (None,None)

        for o in optimizers:
            #Skip learning rates for Adagrad
            if o=='adagrad':
                use_lr = [.001]
            else:
                use_lr = learning_rate

            for lr in use_lr:
                for l in losses:
                    for a in activations:
                        for batch in batch_size:
                            print (o,'|',l,'|',a)
                            print('lr=',lr,'b=',batch)
                            loss= run(data,mean,std,lr,o,a,l,batch,epoch_ct,print_opts='loss',save_model=False)

                            if loss < min_loss:
                                print ('Min loss=',loss,'--')
                                min_loss=loss
                                min_hyperparams=(lr,o,a,l,batch,epoch_ct)

        #Rerun again but this time save model
        lr,o,a,l,batch,epoch = min_hyperparams
        optimum_loss = run(data,mean,std,lr,o,a,l,batch,epoch,print_opts=True,save_model=False)
        print ('Original min_loss, rerun optimum loss=',min_loss,optimum_loss,'diff=',min_loss-optimum_loss)

    else:
        optimizer= 'adagrad'
        learning_rate = .01
        loss = 'mean_squared_error'
        activation = 'leaky_relu'
        batch_size = 200
        epoch_ct=16

        loss = run(data,mean,std,learning_rate,optimizer,activation,
                learning_rate,batch_size,epoch_ct,
                print_opts=True,save_model='model.h5')

        print(loss)

def run(data,mean,std,
    lr,optimizer_type,activation_type,loss_type,
    batch,epoch,
    print_opts='',save_model=False):

    if print_opts==True:
        print_opts='plot,print,loss'

    input_dim = 5
    input_layers=4

    if activation_type=='sigmoid':
        activation ='sigmoid'
    elif activation_type=='tan_h':
        activation='tanh'
    elif activation_type=='relu':
        activation = keras.layers.ReLU(max_value=10000)
    if activation_type=='leaky_relu':
        activation=keras.layers.LeakyReLU(alpha=0.3)

    l_input=Input(shape=(input_dim,),name='Input_Layer')
    l_1=Dense(input_dim,activation=activation,name='Layer_1_')(l_input)
    for n in range(2,input_layers):
        l_1=Dense(input_dim*2,activation=activation,name='Layer_'+str(n)+'_')(l_1)
        # print ('Layer',n)
        # print (l_1)

    #Can't allow the bottom layer to have negative outputs on leaky_relu
    if activation_type=='leaky_relu':
        activation = 'relu'
    l_out=Dense(1,activation=activation,name='Output_Layer')(l_1)

    model=keras.models.Model(l_input,l_out)
    #Adagrad for sparse data according to internet
    if optimizer_type=='adagrad':
        optimizer=keras.optimizers.Adagrad(lr=lr)
    elif optimizer_type=='RMS':
        optimizer=keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=.001, decay=0.0)
    elif optimizer_type=='adam':
        optimizer=keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=.001, decay=0.1, amsgrad=False)

    model.compile(optimizer,loss=loss_type)

    test= int(len(data) * .8)

    # batch_size=200
    # epoch_ct=20

    input_columns=['Temp','Pres','Flow','Process_0','Process_1']
    output_column='Output'

    x=data.loc[:test,input_columns].values
    y=data.loc[:test,output_column].values

    history = model.fit(x=x,y=y,batch_size=batch,epochs=epoch,verbose=0)

    if 'plot' in print_opts:
        plt.plot(history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show(block=False)


    test_x=data.loc[test:,input_columns].values
    test_y=data.loc[test:,output_column].values

    pred=model.predict(x=test_x,batch_size=batch)

    pred=pred.reshape(-1,)

    if 'print' in print_opts:
        output_pred=pd.DataFrame({'Actual':test_y,'Pred':pred})

        print ('Std, Mean=',std,',',mean,'   0=',(mean*std))
        #Renormalize for display
        output_pred = (output_pred + mean) * std
        output_pred['Difference']= (output_pred['Actual'] - output_pred['Pred']).abs()
        pd.set_option('float_format', '{:,.2f}'.format)
        print ('Actual vs Pred:')
        print (output_pred)
        pd.reset_option('float_format')

    loss=sklearn.metrics.mean_squared_error(test_y,pred)

    if 'loss' in print_opts:
        print ('Loss=',loss)

    #Save model
    if save_model!=False:
        model.save(save_model)


    if 'plot' in print_opts:
        plt.show()

    return loss




if __name__=="__main__":
    main()
