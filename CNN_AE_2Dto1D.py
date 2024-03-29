# -*- coding: utf-8 -*-
"""
Convolutional Autoencoder Architecture with K-means clustering on latent space

@author: bailey
"""
import torch
import torch.nn as nn

from torch.optim import lr_scheduler
import os
import numpy as np
import time
# for PCA
from sklearn.decomposition import PCA
import pandas as pd
import pickle

from kmeans_pytorch import kmeans

# for plot publishing
from publish_plots import loss_plot
from publish_plots import spectra_plot
from publish_plots import spectra_plot_difference
from publish_plots import elbow_sil_graph
from publish_plots import Discrete_3D_scatter
from publish_plots import wind_plot
from publish_plots import wind_plot_difference

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# elbow method from https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
# function returns within-cluster-sum of squared errorsWSS score for k values from 1 to kmax

# define a GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU:-1; GPU0: 1; GPU1: 0;

# define a random seed
np.random.seed(123)
torch.manual_seed(123)

import matplotlib.pyplot as plt

# check 
# define a GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU:-1; GPU0: 1; GPU1: 0;


class CNN_AE(nn.Module):
    '''autoencoder with 1D ConvNN'''

    def __init__(self,input_channels, output_channels, kernel_size, 
            stride, padding):

        super(CNN_AE, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_layers = len(input_channels)

        conv_layers = []
        deconv_layers = []

        ################## build the encoder ##################### 
        # conv
        for i in range(self.num_layers):
            conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels = self.input_channels[i], 
                    out_channels = self.output_channels[i],
                    kernel_size = self.kernel_size[i],
                    stride = self.stride[i],
                    padding = self.padding[i],
                    bias = False), 
                nn.LeakyReLU(0.01),
                nn.BatchNorm1d(output_channels[i])
                ))


        ################## build the decoder ##################### 
        # deconv
        for i in range(self.num_layers - 1):
            deconv_layers.append(nn.Sequential(
                nn.ConvTranspose1d(in_channels = self.output_channels[self.num_layers-1-i], 
                    out_channels = self.input_channels[self.num_layers-1-i], 
                    kernel_size = self.kernel_size[self.num_layers-1-i], 
                    stride = self.stride[self.num_layers-1-i], 
                    padding =self.padding[self.num_layers-1-i], 
                    bias = False),
                nn.LeakyReLU(0.01),
                nn.BatchNorm1d(input_channels[self.num_layers-1-i])
                ))

        # deconv - last layer, we don't need batchnorm for the last layer.
        deconv_layers.append(nn.Sequential(
            nn.ConvTranspose1d(
            in_channels = self.output_channels[0], 
            out_channels = self.input_channels[0],
            kernel_size = self.kernel_size[0],
            stride = self.stride[0],
            padding = self.padding[0],
            bias = False),
            nn.LeakyReLU(0.01)))


        # conv and deconv layers
        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        

        self.encoder = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32, LF))
        
        self.decoder = nn.Sequential(
            nn.Linear(LF, 32),
            nn.ReLU(True),
            nn.Linear(32, 128))

        
    def forward(self, x):
        '''
        Args:
        -----
        x: ndarray
            input time series
        
        Returns:
        --------
        x: ndarray
            estimated input time series

        output: input
            classified result
        '''
        
        ############# enter conv encoder #############
        for i in range(self.num_layers):

            x = self.conv_layers[i](x)

        # get the latent features from convolution
        conv_latent_features = x
        
        # flatten
        output = conv_latent_features.view(-1,128)
        
        ############# enter FC-NN encoder #############
        latent_features = self.encoder(output)
         
        ############# enter FC-NN decoder #############
        output = self.decoder(latent_features)
        
        # reshape
        x = output.view(BATCH_SIZE,32,4)
         
        ############ enter conv decoder ##############
        for i in range(self.num_layers):

            x = self.deconv_layers[i](x) 

        return x, latent_features

class ground_motion_data():
    
    def __init__(self,RSN_list):
        self.RSN_list = RSN_list
        
        # define a samle list for train/validate/test
        self.samples = []
    
        # load files
        x=[]
        file_name = 'hurricaneRecords2Dto1Dramp.txt'
        spectra = np.loadtxt(file_name)
        x = np.delete(spectra,0,0)
        num_GM = len(spectra[0,:])
        for GM in range(0,num_GM):
            spec = torch.tensor(x[:,GM],dtype = torch.float32)
            RSN_list[GM] = torch.tensor(spectra[0,GM], dtype=torch.long)
            self.samples.append((spec,RSN_list[GM]))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def train(train_tensor, model, num_epochs, learning_rate, print_every):
    '''
    Train autoencoder.
    Args: 
    -----
    Returns:
    '''
    # define a train/test loss indicator
    print_loss_total = 0.0
    train_loss_list = []

    # define an optimizer / weight decay means a hyperparameter for model parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
        weight_decay=1e-6)
    
    scheduler = lr_scheduler.StepLR(optimizer,step_size=200,gamma=0.9)
    
    min_loss = 10
    
    # run in epochs
    for epoch in range(num_epochs): # 10000 training epochs
    
        for idx, (inputs, targets) in enumerate(train_tensor):
            
            inputs, targets = inputs.cuda(), targets.cuda()

            # First renew the gradient
            optimizer.zero_grad()
            
            #Conv1d expects inputs of shape [batch, channels, features]
            inputs = inputs.unsqueeze(1)
        
            # Second calculate the model outputs
            outputs, output_latent = model(inputs)

            # Third calculate the loss
            loss = (inputs - outputs).pow(2).mean()
    
            # Fourth backpropagation
            loss.backward()
            
            # Weight update 
            optimizer.step()    

            print_loss_total += loss.item()
                
        if epoch % print_every == 0:  # every 100 epochs 
            # print the training loss
            print_loss_mean = print_loss_total / (print_every * len(train_tensor.dataset))
            
            train_loss_list.append(print_loss_mean)
   
            print_loss_total = 0
            print('epoch [{}/{}], loss:{:.6f}'.format(
            epoch + 1, num_epochs, print_loss_mean))           

            if print_loss_mean < min_loss:
                min_loss = print_loss_mean
                save_model(model)
        
        # increase scheduler
        scheduler.step()
            
    # return output
    return train_loss_list

def postprocess(test_tensor):
    # for prediction after having the best model
    
    model = CNN_AE(
        input_channels = [1, 2, 4, 8, 16], 
        output_channels = [2, 4, 8, 16, 32], 
        kernel_size = [18, 12, 10, 8, 6], 
        stride = [2, 2, 2, 2, 2], 
        padding = [0, 0, 0, 0, 0]).cuda()   
    load_model(model)
    
    # define the loss function 
    loss_func = nn.MSELoss()
    
    test_latent_list = []
    test_output_list = []
    test_input_list = []
    test_loss = 0

    for idx, (inputs, targets) in enumerate(test_tensor):
    
        inputs, targets = inputs.cuda(), targets.cuda()
        #inputs, targets = inputs, targets
        
        #Conv1d expects inputs of shape [batch, channels, features]
        inputs = inputs.unsqueeze(1)
        
        # find loss for the test data
        test_outputs, test_output_latent = model(inputs)
        test_loss += loss_func(test_outputs, inputs)
    
        test_latent_list.append((test_output_latent,targets))
        test_output_list.append(test_outputs)
        test_input_list.append(inputs)
        
    test_loss = test_loss / len(test_tensor.dataset)
    
    return test_output_list, test_latent_list, test_input_list


def save_model(model):
    '''
        Save the trained model.
        You can change the model name 
    '''
    torch.save(model.state_dict(), './AE_1DConv_model_GM_epsilon.pt')

def load_model(model):
    '''
        load the trained model.
    '''
    model.load_state_dict(torch.load('./AE_1DConv_model_GM_epsilon.pt'))

def load_file_num(file_name,RSN_list):
    
    file_read = open(file_name,"r")
    
    file_list = [float(line[:-1]) for line in file_read]
    
    # create dictionary
    zip_iterator = zip(RSN_list, file_list)
    file_dictionary = dict(zip_iterator)
    
    return file_dictionary
    
# function to remove whitespace 
def remove(string):
    return string.replace(" ", "") 

# function to remove quotation marks 
def remove2(string):
    return string.replace('"', "") 

def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax+1):
        kmeans = KMeans(n_clusters = k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0
    
        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
      
        sse.append(curr_sse)
    return sse

def get_SSE_and_sil(x):
    
    sil = []
    kmax = 40
    
    # Kmeans Validation
    SSE = calculate_WSS(x,kmax)
    
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax+1):
      kmeans = KMeans(n_clusters = k).fit(x)
      labels = kmeans.labels_
      sil.append(silhouette_score(x, labels, metric = 'euclidean'))
    
    return(SSE,sil) 

if __name__ == '__main__':
    
    # define the neural network parameters
    num_epochs = 10000
    learning_rate = 0.001
    print_every = 10
    LF = 5
    BATCH_SIZE = 16
    ########################################################################### DOWNLOAD DATA
    #  data folder path
    folder_path = './data'
    os.chdir(folder_path) # change directory  
    
    # load files and create dictionaries
    file_name = 'hurricaneID.txt'
    RSN_read = open(file_name,"r")
    RSN_list = [int(line[:-1]) for line in RSN_read]
    
    file_name = 'Ordered_mag_005.txt'
    ID_dict_gt = load_file_num(file_name,RSN_list)
    ID_dict_gt = {key:val for key, val in ID_dict_gt.items() if val > 0}
    
    Dataset = ground_motion_data(RSN_list)
    num_batches =  len(Dataset)// BATCH_SIZE
    
    train_tensor = torch.utils.data.DataLoader(Dataset, batch_size = BATCH_SIZE,
    shuffle=True, num_workers=0,drop_last=True) 
     
    dataiter = iter(train_tensor)
    inputs, label = dataiter.next()
    
    # load scaled spectra data
    scaled_spectra = []
    file_name = 'hurricaneRecords2Dto1Dramp.txt'
    scaled_spectra = np.loadtxt(file_name)
    
    end = len(scaled_spectra-1)
    RSN = scaled_spectra[0]
    spectra = scaled_spectra[1:end]
    spectra = np.transpose(spectra,(1,0))

    # create dictionary
    zip_iterator = zip(RSN, spectra)
    Sa_dictionary = dict(zip_iterator) 
    
    ########################################################################### TRAIN MODEL 
    # change output folder path
    folder_path = './figures'
    try:
        os.makedirs(folder_path)
    except:
        pass
    os.chdir(folder_path) # change directory    
    
    # initialize the autoencoder model
    # .cuda() means putting the model into GPU    
    model = CNN_AE(
        input_channels = [1, 2, 4, 8, 16], 
        output_channels = [2, 4, 8, 16, 32], 
        kernel_size = [18, 12, 10, 8, 6], 
        stride = [2, 2, 2, 2, 2], 
        padding = [0, 0, 0, 0, 0]).cuda()    
    
    start = time.time()
    # train the model
    train_loss_list = train(train_tensor, model, num_epochs, learning_rate,
        print_every)
    end = time.time()    
    
    print("The running time:", (end-start)/60, "min")
    # evaluate the prediction performance
    test_output_list, test_latent_list, test_input_list = postprocess(train_tensor)
    
    # combine all of the batches to single tensor
    test_outputs = test_output_list[0]
    for i in range(num_batches-1):
        test_outputs = torch.cat((test_outputs,test_output_list[i+1]))
        
    test_inputs = test_input_list[0]
    for i in range(num_batches-1):
        test_inputs = torch.cat((test_inputs,test_input_list[i+1]))
        test_latents = test_latent_list[0][0]
    
    test_latents_RSN = test_latent_list[0][1]
    for i in range(num_batches-1):
        test_latents = torch.cat((test_latents,test_latent_list[i+1][0]))
        test_latents_RSN = torch.cat((test_latents_RSN,test_latent_list[i+1][1]))
    test_latent_list = [test_latents,test_latents_RSN]
    
    # save tensors
    torch.save(train_loss_list, './train_loss_list.pt')
    torch.save(train_tensor, './train_tensor.pt')
    torch.save(test_outputs, './test_outputs.pt')
    torch.save(test_inputs, './test_inputs.pt')
    torch.save(test_latent_list, './test_latent_list.pt')
    
    # save dictionaries    
    a_file = open("ID_dict_gt.pkl", "wb")
    pickle.dump(ID_dict_gt, a_file)
    a_file.close()

    a_file = open("Sa_dictionary.pkl", "wb")
    pickle.dump(Sa_dictionary, a_file)
    a_file.close()     
    
    ########################################################################### POST PROCESS AND CREATE PLOTS
    # download tensors
    test_latent_list = torch.load('test_latent_list.pt')
    test_outputs = torch.load('test_outputs.pt')
    test_inputs = torch.load('test_inputs.pt')
    train_loss_list = torch.load('train_loss_list.pt')
    
    test_latents = test_latent_list[0]
    test_latents_RSN = test_latent_list[1]
        
    np_test_outputs = test_outputs.cpu().detach().numpy()
    np_test_inputs = test_inputs.cpu().detach().numpy()
    np_latent_features = test_latents.cpu().detach().numpy()
    np_test_latents_RSN = test_latents_RSN.cpu().detach().numpy()
    
    # download dictionaries
    a_file = open("ID_dict_gt.pkl", "rb")
    ID_dict_gt = pickle.load(a_file)
    a_file.close()
    
    a_file = open("Sa_dictionary.pkl", "rb")
    Sa_dictionary = pickle.load(a_file)
    a_file.close()
    
    ########################################################################### PCA
    
    # Perform PCA on the latent features
    feat_cols = ['feature'+str(i) for i in range(np_latent_features.shape[1])]
    normalised_latents = pd.DataFrame(np_latent_features,columns=feat_cols)
    normalised_latents.tail()
    
    pca_latents = PCA(n_components=3)
    principalComponents_latents = pca_latents.fit_transform(np_latent_features)

    principal_breast_Df = pd.DataFrame(data = principalComponents_latents
             , columns = ['principal component 1', 'principal component 2','principal component 3'])
    
    print('Explained variation per principal component: {}'.format(pca_latents.explained_variance_ratio_))
    
    ########################################################################### K-MEANS Clustering
    x = test_latents.cpu()
    SSE,sil = get_SSE_and_sil(np_latent_features)
    num_clusters = 4
    # k-means cluster
    cluster_ids_x, cluster_centers = kmeans(
        X=x, num_clusters=num_clusters, distance='euclidean', device=torch.device('cpu') ) 
    
    # convert discovered IDs to numpy and correct scaling
    np_cluster_ids_x = cluster_ids_x.tolist()
    np_cluster_ids_x = [x + 1 for x in np_cluster_ids_x]
    
    # create cluster ID dictionary
    zip_iterator = zip(np_test_latents_RSN, np_cluster_ids_x)
    ID_dict_discovered = dict(zip_iterator)
    ########################################################################### CREATE PLOTS
        
    T = np.linspace(0,307,308)
    test_latent_list = torch.load('test_latent_list.pt',map_location=torch.device('cpu'))
    
    # plot training loss
    loss_plot(train_loss_list[1:],print_every)
    
    # plot input spectral groups
    np_test_inputs=np_test_inputs.squeeze()
    wind_plot(np_test_inputs,T,'inputs')
    
    # plot output {reconstructed} spectral groups
    np_test_outputs=np_test_outputs.squeeze()
    wind_plot(np_test_outputs,T,'reconstructed_outputs')
    
    # plot difference from input and output
    wind_plot_difference(np_test_inputs-np_test_outputs,T,'reconstruction_difference')
    
    # plot shilloette score and elbow graph
    elbow_sil_graph(SSE,sil,'elbow_sil_graph')
    
    # Discrete_3D_scatter(test_latents_RSN,ID_dict_gt,np_latent_features,[0,1,2],'Latent Features ','Ground Truth Clusters','LF_3D_gt',num_clusters)
    # Discrete_3D_scatter(test_latents_RSN,ID_dict_gt,principalComponents_latents,[0,1,2],'Principal Comp. ','Ground Truth Clusters','PC_3D_gt',num_clusters)
    # Discrete_3D_scatter(test_latents_RSN,ID_dict_discovered,principalComponents_latents,[0,1,2],'Principal Comp. ','Discovered Clusters','PC_3D_discovered',num_clusters)
    
#%%
import matplotlib.pyplot as plt
small_fig_size = (9,3)
plt_line_width = 0.8
T2 = np.linspace(0,153,154)
for ii in range(len(np_test_inputs)-185):
    fig = plt.figure(figsize=small_fig_size)
    ax = fig.add_subplot(121)
    line_ori,=ax.plot(T2,np_test_inputs[ii,0:len(np_test_inputs[0])//2], linewidth=plt_line_width)    
    line_rec,=ax.plot(T2,np_test_outputs[ii,0:len(np_test_inputs[0])//2], linewidth=plt_line_width)
    ax.legend([line_ori,line_rec],['Original','Reconstructed'],prop={'size': 10})
    ax.set_xlabel('Time [10min]',fontsize=10)
    ax.set_ylabel('Wind Speed in North [m/s]',fontsize=10)
        
    ax = fig.add_subplot(122)
    line_ori,=ax.plot(T2,np_test_inputs[ii,len(np_test_inputs[0])//2:], linewidth=plt_line_width)    
    line_rec,=ax.plot(T2,np_test_outputs[ii,len(np_test_inputs[0])//2:], linewidth=plt_line_width)
    ax.legend([line_ori,line_rec],['Original','Reconstructed'],prop={'size': 10})
    ax.set_xlabel('Time [10min]',fontsize=10)
    ax.set_ylabel('Wind Speed in East [m/s]',fontsize=10)
    plt.rc('xtick', labelsize=9)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=9)    # fontsize of the tick labels
#%%
for ii in range(len(np_test_inputs)-185):
    fig = plt.figure(figsize=small_fig_size)
    ax = fig.add_axes([0, 0, 1, 1])
    line_err,=ax.plot(T,np_test_outputs[ii,:]-np_test_inputs[ii,:], linewidth=plt_line_width)
    ax.set_xlabel('Time [10min]')
    ax.set_ylabel('Wind Speed. [m/s]')                        