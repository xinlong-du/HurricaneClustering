# -*- coding: utf-8 -*-
"""
Simple Autoencoder Architecture with K-means clustering on latent space

@author: bailey
"""

import torch
import torch.nn as nn

from torch.optim import lr_scheduler
import os
import numpy as np
import time
import copy

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
from publish_plots import elbow_graph
from publish_plots import Discrete_3D_scatter
from publish_plots import wind_plot
from publish_plots import wind_plot_difference
from publish_plots import Continuous_3D_scatter

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# elbow method from https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
# function returns within-cluster-sum of squared errorsWSS score for k values from 1 to kmax

# define a GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU:-1; GPU0: 1; GPU1: 0;

class FNN_AE(nn.Module):
    '''
        General Auto-Encoder framework for fully-connected neural networs.    
    '''
    def __init__(self):
        super(FNN_AE, self).__init__()

        ################################ SYNTHETIC DATABASE AARCHITECTURE #####
        input_dim = len(spectra[0])
        # nn.Linear(in_features, out_features)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(), 
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, LF))  
        
        self.decoder = nn.Sequential(
            nn.Linear(LF, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Linear(128, input_dim))
        
    def forward(self, x):
        # x is input
        x_latent = self.encoder(x) # x is the latent space
        x = self.decoder(x_latent)
        
        return x, x_latent
    
class ground_motion_data():
    
    def __init__(self,RSN_list):
        self.RSN_list = RSN_list
        
        # define a samle list for train/validate/test
        self.samples = []
    
        # load files
        x=[]
        file_name = './windRecordsMass/windRecords2Dto1DrampGrid'+str(gridID)+'.txt'
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
    
    model = FNN_AE().cuda()
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
    torch.save(model.state_dict(), './Grid'+str(gridID)+'_AE_FC_model_GM.pt')

def load_model(model):
    '''
        load the trained model.
    '''
    model.load_state_dict(torch.load('./Grid'+str(gridID)+'_AE_FC_model_GM.pt'))
    
def load_file_num(file_name,RSN_list):
    
    file_read = open(file_name,"r")
    
    file_list = [float(line[:-1]) for line in file_read]
    
    # create dictionary
    zip_iterator = zip(RSN_list, file_list)
    file_dictionary = dict(zip_iterator)
    
    return file_dictionary

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
    kmax = 20
    
    # Kmeans Validation
    SSE = calculate_WSS(x,kmax)
    
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax+1):
      kmeans = KMeans(n_clusters = k).fit(x)
      labels = kmeans.labels_
      sil.append(silhouette_score(x, labels, metric = 'euclidean'))
    
    return(SSE,sil)  
#%%    
if __name__ == '__main__':
    
    # define the neural network parameters
    num_epochs = 5000
    learning_rate = 0.001
    print_every = 10
    LF = 5
    BATCH_SIZE = 16
    #%%####################### DOWNLOAD DATA #################################
    for gridID in range(86,87):
        # define a random seed
        np.random.seed(123)
        torch.manual_seed(123)
        #  data folder path
        folder_path = 'C:\\Users\\xinlo\\OneDrive - Northeastern University\\CRISP\\Repositories\\Hurricane Catalog\\HurricaneClustering\\data'
        os.chdir(folder_path) # change directory  
    
        # load files and create dictionaries
        file_name = './windRecordsMass/hurricaneIDsGrid'+str(gridID)+'.txt'
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
        file_name = './windRecordsMass/windRecords2Dto1DrampGrid'+str(gridID)+'.txt'
        scaled_spectra = np.loadtxt(file_name)
        
        end = len(scaled_spectra-1)
        RSN = scaled_spectra[0]
        spectra = scaled_spectra[1:end]
        spectra = np.transpose(spectra,(1,0))
    
        # create dictionary
        zip_iterator = zip(RSN, spectra)
        Sa_dictionary = dict(zip_iterator) 
    
        #%%########################### TRAIN MODEL ###############################
        # change output folder path
        folder_path = './figures'
        try:
            os.makedirs(folder_path)
        except:
            pass
        os.chdir(folder_path) # change directory 
    
        # initialize the autoencoder model
        model = FNN_AE().cuda()
    
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
        torch.save(train_loss_list, './Grid'+str(gridID)+'_train_loss_list.pt')
        torch.save(train_tensor, './Grid'+str(gridID)+'_train_tensor.pt')
        torch.save(test_outputs, './Grid'+str(gridID)+'_test_outputs.pt')
        torch.save(test_inputs, './Grid'+str(gridID)+'_test_inputs.pt')
        torch.save(test_latent_list, './Grid'+str(gridID)+'_test_latent_list.pt')
        
        # save dictionaries    
        a_file = open('Grid'+str(gridID)+'_ID_dict_gt.pkl', "wb")
        pickle.dump(ID_dict_gt, a_file)
        a_file.close()
    
        a_file = open('Grid'+str(gridID)+'_Sa_dictionary.pkl', "wb")
        pickle.dump(Sa_dictionary, a_file)
        a_file.close()  
        
        #%%############## POST PROCESS ###########################################
        # download tensors
        test_latent_list = torch.load('Grid'+str(gridID)+'_test_latent_list.pt')
        test_outputs = torch.load('Grid'+str(gridID)+'_test_outputs.pt')
        test_inputs = torch.load('Grid'+str(gridID)+'_test_inputs.pt')
        train_loss_list = torch.load('Grid'+str(gridID)+'_train_loss_list.pt')
        
        test_latents = test_latent_list[0]
        test_latents_RSN = test_latent_list[1]
            
        np_test_outputs = test_outputs.cpu().detach().numpy()
        np_test_inputs = test_inputs.cpu().detach().numpy()
        np_latent_features = test_latents.cpu().detach().numpy()
        np_test_latents_RSN = test_latents_RSN.cpu().detach().numpy()
        
        # download dictionaries
        a_file = open('Grid'+str(gridID)+'_ID_dict_gt.pkl', "rb")
        ID_dict_gt = pickle.load(a_file)
        a_file.close()
        
        a_file = open('Grid'+str(gridID)+'_Sa_dictionary.pkl', "rb")
        Sa_dictionary = pickle.load(a_file)
        a_file.close()
        
        #%%############################# PCA #####################################    
        # Perform PCA on the latent features
        feat_cols = ['feature'+str(i) for i in range(np_latent_features.shape[1])]
        normalised_latents = pd.DataFrame(np_latent_features,columns=feat_cols)
        normalised_latents.tail()
        
        pca_latents = PCA(n_components=3)
        principalComponents_latents = pca_latents.fit_transform(np_latent_features)
    
        principal_breast_Df = pd.DataFrame(data = principalComponents_latents
                 , columns = ['principal component 1', 'principal component 2','principal component 3'])
        
        print('Explained variation per principal component: {}'.format(pca_latents.explained_variance_ratio_))
        
        #%%######################### K-MEANS Clustering ##########################
        x = test_latents.cpu()
        SSE,sil = get_SSE_and_sil(np_latent_features)
        # sil2=copy.deepcopy(sil)
        # sil2[0]=0.0
        # sil2[1]=0.0
        # num_clusters = max(range(len(sil2)), key=sil2.__getitem__)+2
        
        # load scaled spectra data
        num_clusters_grids = []
        file_name = '../../assets/numClusters.txt'
        num_clusters_grids = np.loadtxt(file_name)
        num_clusters = int(num_clusters_grids[gridID-1])
        
        # k-means cluster
        cluster_ids_x, cluster_centers = kmeans(
            X=x, num_clusters=num_clusters, distance='euclidean', device=torch.device('cpu') ) 
        
        # convert discovered IDs to numpy and correct scaling
        np_cluster_ids_x = cluster_ids_x.tolist()
        np_cluster_ids_x = [x + 1 for x in np_cluster_ids_x]
        
        # create cluster ID dictionary
        zip_iterator = zip(np_test_latents_RSN, np_cluster_ids_x)
        ID_dict_discovered = dict(zip_iterator)
        #%%############################ CREATE PLOTS #############################        
        import matplotlib.pyplot as plt
        small_fig_size = (3.5,3.5)
        small_fig_size2 = (3.5,3.0)
        plt_line_width = 0.5
        plt_line_width_thick = 1.5 
        fig_font_size = 8
        T2 = 10.0*np.linspace(0,len(spectra[0])//2-1,len(spectra[0])//2)
        T = 10.0*np.linspace(0,len(spectra[0])-1,len(spectra[0]))
        
        # plot training loss
        loss_plot(train_loss_list[1:],print_every,'Grid'+str(gridID)+'_train_loss')
        
        # plot input spectral groups
        wind_plot(np_test_inputs,T2,'Grid'+str(gridID)+'_inputs')
        
        # plot output {reconstructed} spectral groups
        wind_plot(np_test_outputs,T2,'Grid'+str(gridID)+'_reconstructed_outputs')
        
        # plot difference from input and output
        wind_plot_difference(np_test_inputs-np_test_outputs,T,'Grid'+str(gridID)+'_reconstruction_difference')
        
        # plot shilloette score and elbow graph
        elbow_sil_graph(SSE,sil,'Grid'+str(gridID)+'_elbow_sil_graph')
        elbow_graph(SSE,sil,'Grid'+str(gridID)+'_elbow_graph')
        #%%
        # Discrete_3D_scatter(test_latents_RSN,ID_dict_gt,np_latent_features,[0,1,2],'Latent Features ','Ground Truth Clusters','LF_3D_gt',num_clusters)
        # Discrete_3D_scatter(test_latents_RSN,ID_dict_gt,principalComponents_latents,[0,1,2],'Principal Comp. ','Ground Truth Clusters','PC_3D_gt',num_clusters)
        # Discrete_3D_scatter(test_latents_RSN,ID_dict_discovered,principalComponents_latents,[0,1,2],'Principal Comp. ','Cluster','Grid'+str(gridID)+'_PC_3D',num_clusters)
        color_name = 'rainbow'
        big_fig_size = (3.5,4.5)
        
        dictionary=ID_dict_discovered;
        latents=principalComponents_latents;
        latent_feature_plot=[0,1,2];
        axis_label='Principal Comp. ';
        cb_label='Cluster';
        fig_name='Grid'+str(gridID)+'_PC_3D';
        
        new_LF = []
        static_attribute_list = []
        count = 0
        # remove data without static attribute to not confuse plotting
        for i in range(len(test_latents_RSN)): 
            try:
                att = dictionary[int(test_latents_RSN[i])]
            except KeyError:
                count+= 1
            else:
                if len(static_attribute_list) == 0:
                    new_LF = latents[[i]]
                    static_attribute_list = [dictionary[int(test_latents_RSN[i])]]
                else:  
                    new_LF = np.append(new_LF,[latents[i]],axis = 0)
                    static_attribute_list.append(att)
        
        fig = plt.figure(figsize=big_fig_size)
        ax1 = fig.add_subplot(111,projection='3d')
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        
        ax1.set_xlabel(axis_label+str(latent_feature_plot[0]+1),labelpad=0.1,fontsize=fig_font_size)
        ax1.set_ylabel(axis_label+str(latent_feature_plot[1]+1),labelpad=0.1,fontsize=fig_font_size)
        ax1.set_zlabel(axis_label+str(latent_feature_plot[2]+1),labelpad=0.1,fontsize=fig_font_size)
        ax1.set_xticks(np.arange(-2, 3, 1))
        ax1.set_yticks(np.arange(-2, 3, 1))
        ax1.set_zticks(np.arange(-2, 3, 1))
        ax1.xaxis.set_tick_params(which='major', direction='in', pad=0.01)
        ax1.yaxis.set_tick_params(which='major', direction='in', pad=0.01)
        ax1.zaxis.set_tick_params(which='major', direction='in', pad=0.01)
        
        if num_clusters == 2:
            p = ax1.scatter(
                new_LF[:, latent_feature_plot[0]], new_LF[:, latent_feature_plot[1]], new_LF[:, latent_feature_plot[2]],
                c=static_attribute_list, 
                cmap=plt.cm.get_cmap(color_name, num_clusters),vmin=0.5,vmax=num_clusters+0.5)
        else:
            p = ax1.scatter(
                new_LF[:, latent_feature_plot[0]], new_LF[:, latent_feature_plot[1]], new_LF[:, latent_feature_plot[2]],
                c=static_attribute_list, 
                cmap=plt.cm.get_cmap(color_name, num_clusters),vmin=0.5,vmax=num_clusters+0.5)
        ax1.view_init(25, 7)
        
        cbar=fig.colorbar(p,ticks=range(num_clusters+1), label=cb_label,pad = 0.1, orientation = 'horizontal')
        cbar.set_label(label=cb_label,fontsize=fig_font_size)
          
        plt.tight_layout()
        plt.savefig('./'+fig_name+'.tif', transparent=False, bbox_inches='tight', dpi=400)
        plt.show()
        #%%
        # for ii in range(10):
        #     fig = plt.figure(figsize=small_fig_size)
        #     ax = fig.add_subplot(211)
        #     plt.rc('xtick', labelsize=fig_font_size)    # fontsize of the tick labels
        #     plt.rc('ytick', labelsize=fig_font_size)    # fontsize of the tick labels
        #     ax.tick_params(direction="in")
        #     line_ori,=ax.plot(T2,np_test_inputs[ii,0:len(np_test_inputs[0])//2], linewidth=plt_line_width)    
        #     line_rec,=ax.plot(T2,np_test_outputs[ii,0:len(np_test_inputs[0])//2], linewidth=plt_line_width)
        #     ax.legend([line_ori,line_rec],['Original','Reconstructed'],prop={'size': fig_font_size})
        #     # ax.set_xlabel('Time (min)',fontsize=fig_font_size)
        #     ax.set_ylabel('Wind speed in North dir. (m/s)',fontsize=fig_font_size)
                
        #     ax = fig.add_subplot(212)
        #     plt.rc('xtick', labelsize=fig_font_size)    # fontsize of the tick labels
        #     plt.rc('ytick', labelsize=fig_font_size)    # fontsize of the tick labels
        #     ax.tick_params(direction="in")
        #     line_ori,=ax.plot(T2,np_test_inputs[ii,len(np_test_inputs[0])//2:], linewidth=plt_line_width)    
        #     line_rec,=ax.plot(T2,np_test_outputs[ii,len(np_test_inputs[0])//2:], linewidth=plt_line_width)
        #     ax.legend([line_ori,line_rec],['Original','Reconstructed'],prop={'size': fig_font_size})
        #     ax.set_xlabel('Time (min)',fontsize=fig_font_size)
        #     ax.set_ylabel('Wind speed in East dir. (m/s)',fontsize=fig_font_size)
        #     plt.rc('xtick', labelsize=fig_font_size)    # fontsize of the tick labels
        #     plt.rc('ytick', labelsize=fig_font_size)    # fontsize of the tick labels
        
        # for ii in range(10):
        #     fig = plt.figure(figsize=small_fig_size)
        #     ax = fig.add_axes([0, 0, 1, 1])
        #     plt.rc('xtick', labelsize=fig_font_size)    # fontsize of the tick labels
        #     plt.rc('ytick', labelsize=fig_font_size)    # fontsize of the tick labels
        #     ax.tick_params(direction="in")
        #     line_err,=ax.plot(T,np_test_outputs[ii,:]-np_test_inputs[ii,:], linewidth=plt_line_width)
        #     ax.set_xlabel('Time (min)',fontsize=fig_font_size)
        #     ax.set_ylabel('Wind speed difference (m/s)',fontsize=fig_font_size)
        #%% plot error
        error=abs(np_test_inputs-np_test_outputs)
        error=error.flatten() 
        weights = np.ones_like(error) / len(error)
        fig = plt.figure(figsize=small_fig_size2)
        ax = fig.add_axes([0, 0, 1, 1])
        plt.rc('xtick', labelsize=fig_font_size)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=fig_font_size)    # fontsize of the tick labels
        ax.tick_params(direction="in")
        (nErr, binsErr, patchesErr) = plt.hist(error, weights=weights, bins=26, facecolor="None", edgecolor="k")  # density=False would make counts
        plt.ylabel('Probability',fontsize=fig_font_size)
        plt.xlabel('Reconstruction error (m/s)',fontsize=fig_font_size);
        ax.set_xlim(0.0, 6)
        file_name = 'Grid'+str(gridID)+'_error_hist'
        plt.savefig('./'+file_name+'.tif', transparent=False, bbox_inches='tight', dpi=300)                   
        # #%% plot latent features clustered using K-means
        # Continuous_3D_scatter(np_test_latents_RSN,ID_dict_discovered,np_latent_features,[0,1,2],'LF','LF','Grid'+str(gridID)+'_LF123')
        # Continuous_3D_scatter(np_test_latents_RSN,ID_dict_discovered,np_latent_features,[2,3,4],'LF','LF','Grid'+str(gridID)+'_LF345')
        
        # #%% plot principal components clusted using k-means
        # Continuous_3D_scatter(np_test_latents_RSN,ID_dict_discovered,principalComponents_latents,[0,1,2],'PC','Cluster','Grid'+str(gridID)+'_PC123')
        #%% select records from each cluster based on the distance to the centroid
        cluster_list=[[] for _ in range(num_clusters)]
        for i in range(len(test_latents_RSN)):
            for ii in range(num_clusters):
                if ID_dict_discovered[int(test_latents_RSN[i])]==ii+1:
                    cluster_list[ii].append(int(test_latents_RSN[i]))
        
        np_cluster_centers = cluster_centers.cpu().detach().numpy()
        np_distances = np.zeros(shape=(len(np_latent_features),num_clusters))
        for i in range(num_clusters):
            for j in range(len(np_latent_features)):
                np_distances[j,i] = np.linalg.norm(np_cluster_centers[i,:]-np_latent_features[j,:])
        
        clusters_distance = []
        for i in range(num_clusters):
            one_cluster_dist = np.zeros(shape=(len(cluster_list[i]),2))
            for j in range(len(cluster_list[i])):
                one_cluster_dist[j,0] = cluster_list[i][j]
                idx = np.where(np_test_latents_RSN == cluster_list[i][j])
                one_cluster_dist[j,1] = np_distances[idx[0][0],i]
            one_cluster_dist_sorted = one_cluster_dist[one_cluster_dist[:, 1].argsort()]
            clusters_distance.append(one_cluster_dist_sorted)
            
        cluster_list_sele=[[] for _ in range(num_clusters)]
        for i in range(num_clusters):
            nSele=round(len(cluster_list[i])/10)
            for j in range(nSele):
                cluster_list_sele[i].append(int(clusters_distance[i][j,0]))
                
        #%% plot wind records for different clusters
        for i in range(num_clusters):
            fig = plt.figure(figsize=small_fig_size)
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            plt.rc('xtick', labelsize=fig_font_size)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=fig_font_size)    # fontsize of the tick labels
            ax1.tick_params(direction="in")
            ax2.tick_params(direction="in")
            for ii in cluster_list[i]:
                if ii in cluster_list_sele[i]:
                    line_ori,=ax1.plot(T2,spectra[ii-1,0:len(spectra[0])//2], linewidth=plt_line_width_thick)
                else:
                    line_ori,=ax1.plot(T2,spectra[ii-1,0:len(spectra[0])//2], linewidth=plt_line_width, linestyle='dashed')    
                # ax1.set_xlabel('Time (min)',fontsize=fig_font_size)
                ax1.set_ylabel('Wind speed in North dir. (m/s)',fontsize=fig_font_size)
                
                if ii in cluster_list_sele[i]:
                    line_ori,=ax2.plot(T2,spectra[ii-1,len(spectra[0])//2:], linewidth=plt_line_width_thick)
                else:
                    line_ori,=ax2.plot(T2,spectra[ii-1,len(spectra[0])//2:], linewidth=plt_line_width, linestyle='dashed')
                ax2.set_xlabel('Time (min)',fontsize=fig_font_size)
                ax2.set_ylabel('Wind speed in East dir. (m/s)',fontsize=fig_font_size)
            plt.savefig('./Grid'+str(gridID)+'Cluster'+str(i+1)+'windRecords.tif', transparent=False, bbox_inches='tight', dpi=400)

        #%% plot all selected records for different clusters
        color_list=['b','g','r','c','m','y','k','tab:brown']
        fig = plt.figure(figsize=small_fig_size)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        #ax1.set_xlim(0.0, 1500.0)
        #ax2.set_xlim(0.0, 1500.0)
        plt.rc('xtick', labelsize=fig_font_size)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=fig_font_size)    # fontsize of the tick labels
        ax1.tick_params(direction="in")
        ax2.tick_params(direction="in")
        for i in range(num_clusters):
            for ii in cluster_list_sele[i]:
                line_ori,=ax1.plot(T2,spectra[ii-1,0:len(spectra[0])//2], linewidth=plt_line_width, color=color_list[i])    
                # ax1.set_xlabel('Time (min)',fontsize=fig_font_size)
                ax1.set_ylabel('Wind speed in North dir. (m/s)',fontsize=fig_font_size)
                
                line_ori,=ax2.plot(T2,spectra[ii-1,len(spectra[0])//2:], linewidth=plt_line_width, color=color_list[i])
                ax2.set_xlabel('Time (min)',fontsize=fig_font_size)
                ax2.set_ylabel('Wind speed in East dir. (m/s)',fontsize=fig_font_size)
        plt.savefig('../seleRecordsPlot/Grid'+str(gridID)+'seleRecords.tif', transparent=False, bbox_inches='tight', dpi=400)
        #%% save the clusters
        # change output folder path
        folder_path = '../windRecordsMass'
        try:
            os.makedirs(folder_path)
        except:
            pass
        os.chdir(folder_path) # change directory 
    
        try:
            os.remove('clusterListGrid'+str(gridID)+'.txt')
        except:
            pass
        for i in range(num_clusters):
            with open('clusterListGrid'+str(gridID)+'.txt', 'a') as f:
                for item in cluster_list[i]:
                    f.write("%s\t" % item)
                f.write("\n")
        
        try:
            os.remove('clusterListSeleGrid'+str(gridID)+'.txt')
        except:
            pass
        for i in range(num_clusters):
            with open('clusterListSeleGrid'+str(gridID)+'.txt', 'a') as f:
                for item in cluster_list_sele[i]:
                    f.write("%s\t" % item)
                f.write("\n")