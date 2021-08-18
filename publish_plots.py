import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np
from matplotlib.animation import FuncAnimation

color_name = 'rainbow'
cmap = mpl.cm.rainbow

# set the font parameters
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 1
tick_width = 1
tick_mj_sz = 5
tick_mn_sz = 2
plt_line_width = 0.8

small_fig_size = (4,3)
big_fig_size = (6,5)

def loss_plot(x,print_every):

    fig = plt.figure(figsize=small_fig_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    # Set the axis limits
    ax.set_yscale('log')
    # plot
    ax.plot(x, linewidth=plt_line_width)
    # Add the x and y-axis labels
    ax.set_xlabel('Number of Epochs x' + str(print_every))
    ax.set_ylabel('Training Loss')
    # Save figure
    plt.savefig('./train_loss.svg', transparent=False, bbox_inches='tight')

def spectra_plot(spectra,T,file_name):
    fig = plt.figure(figsize=small_fig_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    # Set the axis limits
    ax.set_xscale('log')
    ax.set_xlim(0.05, 10)
    ax.set_ylim(-12, 4)
    # Edit the major and minor tick locations
    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
    ax.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.5))
    # plot
    for ii in range(len(spectra)):
        ax.plot(T,spectra[ii,:], linewidth=plt_line_width)
    # Add the x and y-axis labels
    ax.set_xlabel('Period [seconds]')
    ax.set_ylabel('Spectral Accel. [g]')
    # Save figure
    plt.savefig('./'+file_name+'.svg', transparent=True, bbox_inches='tight')

def spectra_plot_difference(spectra,T,file_name):
    fig = plt.figure(figsize=small_fig_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    # Set the axis limits
    ax.set_xscale('log')
    ax.set_xlim(0.05, 10)
    ax.set_ylim(-2.5, 2.5)
    # Edit the major and minor tick locations
    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
    ax.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
    # plot
    for ii in range(len(spectra)):
        ax.plot(T,spectra[ii,:], linewidth=plt_line_width)
    # Add the x and y-axis labels
    ax.set_xlabel('Period [seconds]')
    ax.set_ylabel('Spectral Accel. [g]')
    # Save figure
    plt.savefig('./'+file_name+'.svg', transparent=True, bbox_inches='tight')  
    
def spectra_plot_comp(spectra_i,spectra_o,T,file_name):
    fig = plt.figure(figsize=small_fig_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    # Set the axis limits
    ax.set_xscale('log')
    ax.set_xlim(0.05, 10)
    ax.set_ylim(-12, 4)
    # Edit the major and minor tick locations
    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
    ax.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.5))
    # plot
    ax.plot(T,spectra_i,'-g',linewidth=plt_line_width,label='Ground Truth')
    ax.plot(T,spectra_o,'--c',linewidth=plt_line_width,label='Reconstruction')
    # Add the x and y-axis labels
    ax.set_xlabel('Period [seconds]')
    ax.set_ylabel('Spectral Accel. [g]')
    # Add legend to plot
    ax.legend(loc='lower left', frameon=False)
    # Save figure
    plt.savefig('./'+file_name+'.svg', transparent=True, bbox_inches='tight')

def Continuous_3D_scatter(test_latents_RSN,dictionary,latents,latent_feature_plot,axis_label,cb_label,fig_name):
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

    fig = plt.figure(figsize=(big_fig_size))
    ax1 = fig.add_subplot(111,projection='3d')
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    
    ax1.set_xlabel(axis_label+str(latent_feature_plot[0]+1),labelpad = 10)
    ax1.set_ylabel(axis_label+str(latent_feature_plot[1]+1),labelpad = 10)
    ax1.set_zlabel(axis_label+str(latent_feature_plot[2]+1),labelpad = 10)
    
    p = ax1.scatter(
        new_LF[:, latent_feature_plot[0]], new_LF[:, latent_feature_plot[1]], new_LF[:, latent_feature_plot[2]],
        c=static_attribute_list, 
        cmap=cmap)
    
    fig.colorbar(p,label=cb_label,pad = 0.2)
    
    plt.tight_layout()
    plt.savefig('./'+fig_name+'.svg', transparent=True, bbox_inches='tight')
    plt.show()
    
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111,projection='3d')
    fig.colorbar(p, label=cb_label,pad = 0.2)
    
    def init():
        ax1.scatter(
            new_LF[:, latent_feature_plot[0]], new_LF[:, latent_feature_plot[1]], new_LF[:, latent_feature_plot[2]],
            c=static_attribute_list, 
            cmap=cmap)
        ax1.set_xlabel(axis_label+str(latent_feature_plot[0]+1),labelpad = 10)
        ax1.set_ylabel(axis_label+str(latent_feature_plot[1]+1),labelpad = 10)
        ax1.set_zlabel(axis_label+str(latent_feature_plot[2]+1),labelpad = 10)
        
        return fig,
    
    def animate(i):
        ax1.view_init(elev=20., azim=0.2*i)
        ax1.grid(False)
        return fig,
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=np.arange(0, 1080, 2), repeat=True)
    anim.save('./'+fig_name+'.gif', dpi=80, writer='PillowWriter', fps=48)    

def input_discovered_clusters(test_latents_RSN,dictionary,latents,latent_feature_plot,axis_label,cb_label,fig_name,num_clusters,axis_list):
    # sort the dictionary by key 
    # new_dict = dict(sorted(dictionary.items()))
    
    N = len(test_latents_RSN)
    L = np.linspace(1,num_clusters,num_clusters)
    convert_matrix = np.zeros([2,num_clusters])
    convert_matrix[0,:] = L

    LF_group = []

    count = 0
    # remove data without static attribute to not confuse plotting
    for i in range(num_clusters):
        for j in range(N):
            try:
                att = dictionary[int(test_latents_RSN[j])]
            except KeyError:
                count+= 1
            else:
                if att == i+1:
                    if len(LF_group) == 0:
                        LF_group = latents[[j]]
                    else:
                        LF_group = np.append(LF_group,[latents[j]],axis = 0)    
        
        print(LF_group)
        
        fig = plt.figure(figsize=big_fig_size)
        ax1 = fig.add_subplot(111,projection='3d')
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        
        ax1.set_xlabel(axis_label+str(latent_feature_plot[0]+1),labelpad=10)
        ax1.set_ylabel(axis_label+str(latent_feature_plot[1]+1),labelpad=10)
        ax1.set_zlabel(axis_label+str(latent_feature_plot[2]+1),labelpad=10)
        
        xmin = axis_list[0]
        xmax = axis_list[1]
        ymin = axis_list[2]
        ymax = axis_list[3]
        zmin = axis_list[4]
        zmax = axis_list[5]
        
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)
        ax1.set_zlim(zmin, zmax)
        
        ax1.scatter(LF_group[:, latent_feature_plot[0]], LF_group[:, latent_feature_plot[1]], LF_group[:, latent_feature_plot[2]])
        
        plt.tight_layout()
        plt.show()
    
        c = input('Enter the cluster classification: ')
        
        convert_matrix[1,i] = c
        LF_group = []
        
    print('covert_matrix')
    print(convert_matrix)
    list2 = []
    # use the conversion matrix 
    for i in range(N):
        rsn = int(test_latents_RSN[i])
        if dictionary[rsn] == 1:
            list2.append(convert_matrix[1,0])
        elif dictionary[rsn] == 2:
            list2.append(convert_matrix[1,1])
        elif dictionary[rsn] == 3:
            list2.append(convert_matrix[1,2])
        elif dictionary[rsn] == 4:
            list2.append(convert_matrix[1,3])
        elif dictionary[rsn] == 5:
            list2.append(convert_matrix[1,4])
            
    zip_iterator = zip(test_latents_RSN, list2)
    dictionary2 = dict(zip_iterator)
                
    
    return dictionary2, convert_matrix
    
def Discrete_3D_scatter(test_latents_RSN,dictionary,latents,latent_feature_plot,axis_label,cb_label,fig_name,num_clusters):
    
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
    
    ax1.set_xlabel(axis_label+str(latent_feature_plot[0]+1),labelpad=10)
    ax1.set_ylabel(axis_label+str(latent_feature_plot[1]+1),labelpad=10)
    ax1.set_zlabel(axis_label+str(latent_feature_plot[2]+1),labelpad=10)
    
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
        
    fig.colorbar(p,ticks=range(num_clusters+1), label=cb_label,pad = 0.2)
      
    plt.tight_layout()
    plt.savefig('./'+fig_name+'.svg', transparent=True, bbox_inches='tight')
    plt.show()
    
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    zmin, zmax = ax1.get_zlim()
    
    axis_list = [xmin,xmax,ymin,ymax,zmin,zmax]
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111,projection='3d')
    fig.colorbar(p,ticks=range(num_clusters+1), label=cb_label,pad = 0.2)
    
    def init():
        if num_clusters == 2:
            ax1.scatter(
                new_LF[:, latent_feature_plot[0]], new_LF[:, latent_feature_plot[1]], new_LF[:, latent_feature_plot[2]],
                c=static_attribute_list, 
                cmap=plt.cm.get_cmap(color_name, num_clusters),vmin=0.5,vmax=num_clusters+0.5)
        else:
            ax1.scatter(
                new_LF[:, latent_feature_plot[0]], new_LF[:, latent_feature_plot[1]], new_LF[:, latent_feature_plot[2]],
                c=static_attribute_list, 
                cmap=plt.cm.get_cmap(color_name, num_clusters),vmin=0.5,vmax=num_clusters+0.5)
        ax1.set_xlabel(axis_label+str(latent_feature_plot[0]+1),labelpad=10)
        ax1.set_ylabel(axis_label+str(latent_feature_plot[1]+1),labelpad=10)
        ax1.set_zlabel(axis_label+str(latent_feature_plot[2]+1),labelpad=10)
        return fig,
    
    def animate(i):
        ax1.view_init(elev=20., azim=0.2*i)
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        ax1.grid(False)
        return fig,
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=np.arange(0, 1080, 2), repeat=True)
    anim.save('./'+fig_name+'.gif', dpi=80, writer='PillowWriter', fps=48) 

    return axis_list     

def elbow_sil_graph(SSE,sil,file_name):
    fig = plt.figure(figsize=(8, 3))
    
    # plot elbow graph
    ax = fig.add_subplot(121)
    ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.plot(SSE,'-g',linewidth=plt_line_width,label='Ground Truth')
    ax.set_xlabel('k value')
    ax.set_ylabel('WSS')
    
    # plot silhouette graph
    ax = fig.add_subplot(122)
    ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.plot(sil,'--c',linewidth=plt_line_width,label='Reconstruction')
    # Add the x and y-axis labels
    ax.set_xlabel('k value')
    ax.set_ylabel('Silhouette Score')
    # Save figure
    plt.tight_layout()
    plt.savefig('./'+file_name+'.svg', transparent=True, bbox_inches='tight')
    
def elbow_sil_graph_synthetic(SSE,sil,file_name):
    fig = plt.figure(figsize=(8, 3))
    
    # plot elbow graph
    ax = fig.add_subplot(121)
    ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.plot(SSE,'-g',linewidth=plt_line_width,label='Ground Truth')
    ax.text(3,SSE[3],'k value = 4')
    ax.plot(3,SSE[3],'or')
    ax.set_xlabel('k value')
    ax.set_ylabel('WSS')
    
    # plot silhouette graph
    ax = fig.add_subplot(122)
    ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.plot(sil,'--c',linewidth=plt_line_width,label='Reconstruction')
    ax.plot(3,sil[3],'or')
    ax.text(3,sil[3],'k value = 4')
    # Add the x and y-axis labels
    ax.set_xlabel('k value')
    ax.set_ylabel('Silhouette Score')
    # Save figure
    plt.tight_layout()
    plt.savefig('./'+file_name+'.svg', transparent=True, bbox_inches='tight')

def centroid_and_mean_2(num_clusters,cluster_ids_x,cluster_centers,test_latents,test_latents_RSN,Sa_dictionary,T,file_name):
    GMNum = len(test_latents_RSN)
    scaled_spectra_list = []
    test_latents = test_latents.cpu()
    # create the figure
    rsn_centroids = [1,1]
    for i in range(num_clusters):
        dis = 10000
        spectra_cluster_list = []
        
        for j in range(GMNum-1):
            ids = cluster_ids_x.numpy()[j] 
            if ids == i:
                # look up ground motion
                GM_RSN = int(test_latents_RSN[j])            
                # plot each cluster's spectra                    
                spectra_cluster_list.append(Sa_dictionary[GM_RSN])
            
            dis_new = sum(abs((cluster_centers[i]-test_latents[j,:]).detach().numpy()))
            if dis_new < dis:
                dis = dis_new
                rsn_centroids[i] = int(test_latents_RSN[j])
                
        scaled_spectra_list.append(spectra_cluster_list)
            
    spectra_cluster_centroid = []
    for i in range(num_clusters):
        spectra_cluster_centroid.append(Sa_dictionary[rsn_centroids[i]])
        
    # plot average for each scaled spectral clusters
    # plot group 1
    fig = plt.figure(figsize=small_fig_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    # Set the axis limits
    ax.set_xscale('log')
    ax.set_xlim(0.05, 10)
    ax.set_ylim(-12, 4)
    # Edit the major and minor tick locations
    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
    ax.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.5))
    
    # plot
    ax.plot(T,spectra_cluster_centroid[0],'-g',linewidth=plt_line_width,label='Cluster Centroid')
    ax.plot(T,average(scaled_spectra_list[0]),'--c',linewidth=plt_line_width,label='Mean Spectrum')
    # Add the x and y-axis labels
    ax.set_xlabel('Period [seconds]')
    ax.set_ylabel('Spectral Accel. [g]')
    # Add legend to plot
    ax.legend(loc='lower left', frameon=False)
    # Save figure
    plt.savefig('./'+file_name+'_1.svg', transparent=True, bbox_inches='tight')
    
    # plot group 2
    fig = plt.figure(figsize=small_fig_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    # Set the axis limits
    ax.set_xscale('log')
    ax.set_xlim(0.05, 10)
    ax.set_ylim(-12, 4)
    # Edit the major and minor tick locations
    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
    ax.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.5))
    
    # plot
    ax.plot(T,spectra_cluster_centroid[1],'-g',linewidth=plt_line_width,label='Cluster Centroid')
    ax.plot(T,average(scaled_spectra_list[1]),'--c',linewidth=plt_line_width,label='Mean Spectrum')
    # Add the x and y-axis labels
    ax.set_xlabel('Period [seconds]')
    ax.set_ylabel('Spectral Accel. [g]')
    # Add legend to plot
    ax.legend(loc='lower left', frameon=False)
    # Save figure
    plt.savefig('./'+file_name+'_2.svg', transparent=True, bbox_inches='tight')    
    
    
    fig = plt.figure(figsize=(8, 3))
    
    # plot group 1
    ax = fig.add_subplot(121)
    ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    
    # Set the axis limits
    ax.set_xscale('log')
    ax.set_xlim(0.05, 10)
    ax.set_ylim(-12, 4)
    # Edit the major and minor tick locations
    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
    ax.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.5))
    
    ax.plot(T,spectra_cluster_centroid[0],'-g',linewidth=plt_line_width,label='Cluster Centroid')
    ax.plot(T,average(scaled_spectra_list[0]),'--c',linewidth=plt_line_width,label='Mean Spectrum')
    
    # Add the x and y-axis labels
    ax.set_xlabel('Period [seconds]')
    ax.set_ylabel('Spectral Accel. [g]')
    # Add legend to plot
    ax.legend(loc='lower left', frameon=False)
    
    # plot group 2
    ax = fig.add_subplot(122)
    ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    
    # Set the axis limits
    ax.set_xscale('log')
    ax.set_xlim(0.05, 10)
    ax.set_ylim(-12, 4)
    # Edit the major and minor tick locations
    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
    ax.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.5))
    
    ax.plot(T,spectra_cluster_centroid[1],'-g',linewidth=plt_line_width,label='Cluster Centroid')
    ax.plot(T,average(scaled_spectra_list[1]),'--c',linewidth=plt_line_width,label='Mean Spectrum')
    # Add the x and y-axis labels
    ax.set_xlabel('Period [seconds]')
    ax.set_ylabel('Spectral Accel. [g]')
    # Add legend to plot
    ax.legend(loc='lower left', frameon=False)
    # Save figure
    plt.tight_layout()
    plt.savefig('./'+file_name+'.svg', transparent=True, bbox_inches='tight')
    
def centroid_and_mean_5(num_clusters,cluster_ids_x,cluster_centers,test_latents,test_latents_RSN,Sa_dictionary,un_Sa_dictionary,T,file_name):
    GMNum = len(test_latents_RSN)
    scaled_spectra_list = []
    unscaled_spectra_list = []
    test_latents = test_latents.cpu()
    # create the figure
    rsn_centroids = [1,1,1,1,1]
    for i in range(num_clusters):
        dis = 10000
        spectra_cluster_list = []
        unspectra_cluster_list = []
        for j in range(GMNum-1):
            ids = cluster_ids_x.numpy()[j] 
            if ids == i:
                # look up ground motion
                GM_RSN = int(test_latents_RSN[j])            
                # plot each cluster's spectra                    
                spectra_cluster_list.append(Sa_dictionary[GM_RSN])
                unspectra_cluster_list.append(un_Sa_dictionary[GM_RSN])
            
            dis_new = sum(abs((cluster_centers[i]-test_latents[j,:]).detach().numpy()))
            if dis_new < dis:
                dis = dis_new
                rsn_centroids[i] = int(test_latents_RSN[j])
                
        scaled_spectra_list.append(spectra_cluster_list)
        unscaled_spectra_list.append(unspectra_cluster_list)
            
    spectra_cluster_centroid = []
    un_spetra_cluster_centroid = []
    for i in range(num_clusters):
        spectra_cluster_centroid.append(Sa_dictionary[rsn_centroids[i]])
        un_spetra_cluster_centroid.append(un_Sa_dictionary[rsn_centroids[i]])
    
    # plot average for each scaled spectral clusters
    fig = plt.figure(figsize=small_fig_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    # Set the axis limits
    ax.set_xscale('log')
    ax.set_xlim(0.05, 10)
    ax.set_ylim(-12, 4)
    # Edit the major and minor tick locations
    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
    ax.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.5))
    rainbow = cm.get_cmap('rainbow', 5)
    for i in range(num_clusters):
        line_color = rainbow(i)
        # ax.plot(T,spectra_cluster_centroid[i],'--',c=line_color,linewidth=plt_line_width)
        ax.plot(T,average(scaled_spectra_list[i]),'-',c=line_color,linewidth=plt_line_width,label='Cluster '+str(i+1))
    # Add the x and y-axis labels
    ax.set_xlabel('Period [seconds]')
    ax.set_ylabel('Spectral Accel. [g]')
    # Add legend to plot
    ax.legend(loc='lower left', frameon=False)
    # Save figure
    plt.tight_layout()
    plt.savefig('./'+file_name+'_scaled.svg', transparent=True, bbox_inches='tight')
    
    # plot average for each unscaled spectral clusters
    fig = plt.figure(figsize=small_fig_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    # Set the axis limits
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.05, 10)
    ax.set_ylim(0.000001, 15)
    # Edit the major and minor tick locations
    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
    ax.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator(x_minor)
    y_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
    ax.yaxis.set_major_locator(y_major)
    y_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.yaxis.set_minor_locator(y_minor)
    
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    rainbow = cm.get_cmap('rainbow', 5)
    for i in range(num_clusters):
        line_color = rainbow(i)
        # ax.plot(T,un_spetra_cluster_centroid[i],'--',c=line_color,linewidth=plt_line_width)
        ax.plot(T,average(unscaled_spectra_list[i]),'-',c=line_color,linewidth=plt_line_width,label='Cluster '+str(i+1))
    # Add the x and y-axis labels
    ax.set_xlabel('Period [seconds]')
    ax.set_ylabel('Spectral Accel. [g]')
    # Add legend to plot
    ax.legend(loc='lower left', frameon=False)
    # Save figure
    plt.tight_layout()
    plt.savefig('./'+file_name+'_unscaled.svg', transparent=True, bbox_inches='tight')
    
def average(list_name):
    return sum(list_name) / len(list_name)

def plot_encoded_2clusters(num_clusters,cluster_ids_x,test_latents_RSN,Group_dictionary,Sa_dictionary,T,file_name):
    GMNum = len(test_latents_RSN)
    scaled_spectra_list = []
    correct = 0
    count = 0
    
    # create the figure
    
    for i in range(num_clusters):
        spectra_cluster_list = []
        fig = plt.figure(figsize=small_fig_size)
        ax = fig.add_axes([0, 0, 1, 1])
        
        ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
        ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
        ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
        ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
        # Set the axis limits
        ax.set_xscale('log')
        ax.set_xlim(0.05, 10)
        ax.set_ylim(-12, 4)
        
        # Edit the major and minor tick locations
        x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
        ax.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.5))
        
        for j in range(GMNum-1):
            ids = cluster_ids_x.numpy()[j] 
            if ids == i:
                # look up ground motion
                GM_RSN = int(test_latents_RSN[j])
                try:
                    c = Group_dictionary[GM_RSN]
                except:
                    count += 1
                    continue
                if c == 1:
                    c_group = 'red'
                elif c == 2:
                    c_group = 'purple'
##################### assign the groups to the clusters manualy##########################
                if c  == 1 and ids == 1:
                    correct += 1
                elif c == 2 and ids == 0:
                    correct += 1               
                # plot each cluster's spectra                    
                spectra_cluster_list.append(Sa_dictionary[GM_RSN])
                ax.plot(T,Sa_dictionary[GM_RSN],color=c_group,linewidth=plt_line_width)
      
        ax.set_xlabel('Period [seconds]')
        ax.set_ylabel('Spectral Accel. [g]')
        # Add legend to plot
        
        scaled_spectra_list.append(spectra_cluster_list)
        plt.savefig('./'+file_name+str(i+1)+'.svg', transparent=True, bbox_inches='tight')
        
    score = correct/GMNum
    print('The AEs classification score is : '+str(score))
    
    # plot average for each scaled spectral clusters
    fig = plt.figure(figsize=small_fig_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    # Set the axis limits
    ax.set_xscale('log')
    ax.set_xlim(0.05, 10)
    ax.set_ylim(-12, 4)
    # Edit the major and minor tick locations
    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
    ax.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.5))
    # plot
    ax.plot(T,average(scaled_spectra_list[0]),'-g',linewidth=plt_line_width,label='Cluster 1 Avg.')
    ax.plot(T,average(scaled_spectra_list[1]),'--c',linewidth=plt_line_width,label='Cluster 2 Avg.')
    # Add the x and y-axis labels
    ax.set_xlabel('Period [seconds]')
    ax.set_ylabel('Spectral Accel. [g]')
    # Add legend to plot
    ax.legend(loc='lower left', frameon=False)
    # Save figure
    plt.savefig('./'+file_name+'averages.svg', transparent=True, bbox_inches='tight')

    # plot average for each scaled spectral clusters
    fig = plt.figure(figsize=small_fig_size)
    # plot
    ax.plot(T,average(scaled_spectra_list[0]),'-c',linewidth=plt_line_width*2,label='Cluster 1 Avg.')
    ax.plot(T,)
    
    ax.plot(T,average(scaled_spectra_list[1]),'-g',linewidth=plt_line_width*2,label='Cluster 2 Avg.')
    
    # Add the x and y-axis labels
    ax.set_xlabel('Period [seconds]')
    ax.set_ylabel('Spectral Accel. [g]')
    ax.set_title('The Classification score is '+str(score))
    # Add legend to plot
    # Save figure
    plt.savefig('./'+file_name+'score_in_title.svg', transparent=True, bbox_inches='tight')  
    
def plot_encoded_5clusters(num_clusters,cluster_ids_x,test_latents_RSN,Group_dictionary,Sa_dictionary,T,file_name):
    GMNum = len(test_latents_RSN)
    scaled_spectra_list = []
    correct = 0
    count = 0
    cmap=plt.cm.get_cmap(color_name, 5)
    
    for i in range(num_clusters):
        spectra_cluster_list = []
        fig = plt.figure(figsize=small_fig_size)
        ax = fig.add_axes([0, 0, 1, 1])
        
        ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
        ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
        ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
        ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
        # Set the axis limits
        ax.set_xscale('log')
        ax.set_xlim(0.05, 10)
        ax.set_ylim(-12, 4)
        
        # Edit the major and minor tick locations
        x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
        ax.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.5))
        for j in range(GMNum-1):
            ids = cluster_ids_x.numpy()[j] 
            if ids == i:
                # look up ground motion
                GM_RSN = int(test_latents_RSN[j])
                try:
                    c = Group_dictionary[GM_RSN]
                except:
                    count += 1
                    continue
##################### assign the groups to the clusters manualy##########################
                if c  == 4 and ids == 0:
                    correct += 1
                elif c == 2 and ids == 1:
                    correct += 1
                elif c == 5 and ids == 2:
                    correct += 1
                elif c == 1 and ids == 3:
                    correct += 1
                elif c == 3 and ids == 4:
                    correct += 1              
                # plot each cluster's spectra                    
                spectra_cluster_list.append(Sa_dictionary[GM_RSN])
                ax.plot(T,Sa_dictionary[GM_RSN],color=cmap(c/5-0.1),linewidth=plt_line_width)
      
        ax.set_xlabel('Period [seconds]')
        ax.set_ylabel('Spectral Accel. [g]')
        
        scaled_spectra_list.append(spectra_cluster_list)
        plt.savefig('./'+file_name+str(i+1)+'.svg', transparent=True, bbox_inches='tight')
        
    score = correct/GMNum
    print('The AEs classification score is : '+str(score))
    
    # plot average for each scaled spectral clusters
    fig = plt.figure(figsize=small_fig_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    # Set the axis limits
    ax.set_xscale('log')
    ax.set_xlim(0.05, 10)
    ax.set_ylim(-12, 4)
    # Edit the major and minor tick locations
    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
    ax.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.5))
    # plot
    ax.plot(T,average(scaled_spectra_list[0]),'-g',linewidth=plt_line_width,label='Cluster 1 Avg.')
    ax.plot(T,average(scaled_spectra_list[1]),'--c',linewidth=plt_line_width,label='Cluster 2 Avg.')
    ax.plot(T,average(scaled_spectra_list[2]),'-.m',linewidth=plt_line_width,label='Cluster 3 Avg.')
    ax.plot(T,average(scaled_spectra_list[3]),':b',linewidth=plt_line_width,label='Cluster 4 Avg.')
    # ax.plot(T,average(scaled_spectra_list[4]),'-k',linewidth=plt_line_width,label='Cluster 5 Avg.')
    # Add the x and y-axis labels
    ax.set_xlabel('Period [seconds]')
    ax.set_ylabel('Spectral Accel. [g]')
    # Add legend to plot
    ax.legend(loc='lower left', frameon=False)
    # Save figure
    plt.savefig('./'+file_name+'averages.svg', transparent=True, bbox_inches='tight')
    # plot average for each scaled spectral clusters
    
    fig = plt.figure(figsize=small_fig_size)
    # plot
    ax.plot(T,average(scaled_spectra_list[0]),'-g',linewidth=plt_line_width,label='Cluster 1 Avg.')
    ax.plot(T,average(scaled_spectra_list[1]),'--c',linewidth=plt_line_width,label='Cluster 2 Avg.')
    # Add the x and y-axis labels
    ax.set_xlabel('Period [seconds]')
    ax.set_ylabel('Spectral Accel. [g]')
    ax.set_title('The Classification score is '+str(score))
    # Add legend to plot
    # Save figure
    plt.savefig('./'+file_name+'score_in_title.svg', transparent=True, bbox_inches='tight')  
    
    
def plot_encoded_clusters(num_clusters,cluster_ids_x,test_latents_RSN,Sa_dictionary,T,file_name):
    # plot scaled spectral clusters 
    GMNum = len(test_latents_RSN)
    scaled_spectra_list = []
    for i in range(num_clusters):
        spectra_cluster_list = []
        fig = plt.figure(figsize=small_fig_size)
        ax = fig.add_axes([0, 0, 1, 1])
        
        ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
        ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
        ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
        ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
        # Set the axis limits
        ax.set_xscale('log')
        ax.set_xlim(0.05, 10)
        ax.set_ylim(-12, 4)
        
        # Edit the major and minor tick locations
        x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
        ax.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.5))
        
        for j in range(GMNum):
            ids = cluster_ids_x.numpy()[j] 
            if ids == i:
                # look up ground motion
                GM_RSN = int(test_latents_RSN[j])
                # plot each cluster's spectra                    
                spectra_cluster_list.append(Sa_dictionary[GM_RSN])
                ax.plot(T,Sa_dictionary[GM_RSN],linewidth=plt_line_width)
        scaled_spectra_list.append(spectra_cluster_list)
        
        ax.set_xlabel('Period [seconds]')
        ax.set_ylabel('Spectral Accel. [g]')
        
        plt.savefig('./'+file_name+str(i+1)+'.svg', transparent=True, bbox_inches='tight')  
    
    # plot average for each scaled spectral clusters
    fig = plt.figure(figsize=small_fig_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    # Set the axis limits
    ax.set_xscale('log')
    ax.set_xlim(0.05, 10)
    ax.set_ylim(-12, 4)
    # Edit the major and minor tick locations
    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
    ax.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.5))
    # plot
    line_type = ['-g','--c','-.m',':b','-k']
    for i in range(num_clusters):
        ax.plot(T,average(scaled_spectra_list[i]),line_type[i],linewidth=plt_line_width,label='Cluster '+str(i+1)+' Avg.')

    # Add the x and y-axis labels
    ax.set_xlabel('Period [seconds]')
    ax.set_ylabel('Spectral Accel. [g]')
    # Add legend to plot
    ax.legend(loc='lower left', frameon=False)
    # Save figure
    plt.savefig('./'+file_name+'averages.svg', transparent=True, bbox_inches='tight')

def LF_PC_3D_plot_by_cluster(test_latents_RSN,groud_truth_dictionary,principalComponents_latents,latent_features,latent_feature_plot,fig_name,cluster_IDs_dictionary):
    
    new_PC = []
    groud_truth_cluster = []
    discovered_cluster = []
    count = 0
    # remove data without static attribute to not confuse plotting
    for i in range(len(test_latents_RSN)): 
        try:
            att1 = groud_truth_dictionary[int(test_latents_RSN[i])]
            att2 = cluster_IDs_dictionary[int(test_latents_RSN[i])]
        except KeyError:
            count+= 1
        else:
            if len(groud_truth_cluster) == 0:
                new_PC = principalComponents_latents[[i]]
                groud_truth_cluster = [groud_truth_dictionary[int(test_latents_RSN[i])]]
                discovered_cluster = [cluster_IDs_dictionary[int(test_latents_RSN[i])]]
            else:  
                new_PC = np.append(new_PC,[principalComponents_latents[i]],axis = 0)
                groud_truth_cluster.append(att1)
                discovered_cluster.append(att2)
                
    print('the total number of popped spectra is: '+str(count))
    cmap = mpl.cm.rainbow
    norm = mpl.colors.Normalize(vmin=3, vmax=8)
    colorbar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121,projection='3d')
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    
    ax1.scatter(
        new_PC[:, latent_feature_plot[0]], new_PC[:, latent_feature_plot[1]], new_PC[:, latent_feature_plot[2]],
        c=groud_truth_cluster, 
        cmap=color_name)
    ax1.set_xlabel('Principal Component 1',labelpad = 15)
    ax1.set_ylabel('Principal Component 2',labelpad = 15)
    ax1.set_zlabel('Principal Component 3',labelpad = 15)
    
    ax2 = fig.add_subplot(122,projection='3d')
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    
    ax2.scatter(
        new_PC[:, latent_feature_plot[0]], new_PC[:, latent_feature_plot[1]], new_PC[:, latent_feature_plot[2]],
        c=discovered_cluster, 
        cmap=color_name)
    
    ax2.set_xlabel('Principal Component 1',labelpad = 15)
    ax2.set_ylabel('Principal Component 2',labelpad = 15)
    ax2.set_zlabel('Principal Component 3',labelpad = 15)

    cbar_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
    fig.colorbar(colorbar, cax=cbar_ax, label='Magnitude', shrink = 0.2)
    
    plt.tight_layout()
    plt.savefig('./'+fig_name+'.svg', transparent=True, bbox_inches='tight')
    plt.show()
    num_clusters=2
    def init():
        p1 = ax1.scatter(
            new_PC[:, latent_feature_plot[0]], new_PC[:, latent_feature_plot[1]], new_PC[:, latent_feature_plot[2]],
            c=groud_truth_cluster, 
            cmap=color_name)
        ax1.set_xlabel('Principal Component 1',labelpad = 15)
        ax1.set_ylabel('Principal Component 2',labelpad = 15)
        ax1.set_zlabel('Principal Component 3',labelpad = 15)
        
        fig.colorbar(p1, label='Magnitude', shrink = 0.2,orientation='horizontal')
        
        p2 = ax2.scatter(
            new_PC[:, latent_feature_plot[0]], new_PC[:, latent_feature_plot[1]], new_PC[:, latent_feature_plot[2]],
            c=discovered_cluster, 
            cmap=color_name)
        
        ax2.set_xlabel('Principal Component 1',labelpad = 15)
        ax2.set_ylabel('Principal Component 2',labelpad = 15)
        ax2.set_zlabel('Principal Component 3',labelpad = 15)
        
        fig.colorbar(p2,ticks=range(num_clusters+1), label='Discovered Clusters',pad = 0.2,orientation='horizontal')
        
        return fig,
    
    def animate(i):
        ax1.view_init(elev=20., azim=0.2*i)
        ax2.view_init(elev=20., azim=0.2*i)
        ax1.grid(False)
        ax2.grid(False)
        return fig,
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=np.arange(0, 720, 1), repeat=True)
    anim.save('./'+fig_name+'.gif', dpi=80, writer='PillowWriter', fps=48)
    
def plot_sample_spectra_from_latent_space(test_latents_RSN,ID_dict_discovered,principalComponents_latents,num_clusters,test_latent_list,cluster_ids_x,Sa_dictionary,un_Sa_dictionary,T,file_name):
    new_LF = []
    ID_list = []
    count = 0
    # remove data without static attribute to not confuse plotting
    for i in range(len(test_latents_RSN)): 
        try:
            att = ID_dict_discovered[int(test_latents_RSN[i])]
        except KeyError:
            count+= 1
        else:
            if len(ID_list) == 0:
                new_LF = principalComponents_latents[[i]]
                ID_list = [ID_dict_discovered[int(test_latents_RSN[i])]]
            else:  
                new_LF = np.append(new_LF,[principalComponents_latents[i]],axis = 0)
                ID_list.append(att) 
    
    GMNum = len(cluster_ids_x)
    
    # plot scaled spectral clusters 
    scaled_spectra_list = []
    for i in range(num_clusters):
        spectra_cluster_list = []
        # set plotting parameters
        fig = plt.figure(figsize=small_fig_size)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
        ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
        ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
        ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
        # Set the axis limits
        ax.set_xscale('log')
        ax.set_xlim(0.05, 10)
        ax.set_ylim(-12, 4)
        # Edit the major and minor tick locations
        x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
        ax.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.5))
        # plot
        rainbow = cm.get_cmap('rainbow', 5)
        line_color = rainbow(i)
            
        for j in range(GMNum):
            ids = cluster_ids_x.numpy()[j] 
            if ids == i:
                # look up ground motion
                GM_ID = int(test_latent_list[1][j])
                # plot each cluster's spectra                    
                spectra_cluster_list.append(Sa_dictionary[GM_ID])
                plt.plot(T,Sa_dictionary[GM_ID],c = line_color)
        scaled_spectra_list.append(spectra_cluster_list)
        # Add the x and y-axis labels
        ax.set_xlabel('Period [seconds]')
        ax.set_ylabel('Spectral Accel. [g]')
        # Save figure
        plt.savefig('./scaled_'+file_name+'_'+str(i)+'.svg', transparent=True, bbox_inches='tight')  
        
    # plot average for each scaled spectral clusters 
    # set plotting parameters
    fig = plt.figure(figsize=small_fig_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    # Set the axis limits
    ax.set_xscale('log')
    ax.set_xlim(0.05, 10)
    ax.set_ylim(-12, 4)
    # Edit the major and minor tick locations
    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
    ax.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.5))
    # plot
    line_type = ['-g','--c','-.m',':b','-k']
    for i in range(num_clusters):
        ax.plot(T,average(scaled_spectra_list[i]),line_type[i],linewidth=plt_line_width,label='Cluster '+str(i+1)+' Avg.')
    # Add legend to plot
    ax.legend(loc='lower left', frameon=False)
    # Add the x and y-axis labels
    ax.set_xlabel('Period [seconds]')
    ax.set_ylabel('Spectral Accel. [g]')
    # Save figure
    plt.savefig('./scaled_means_'+file_name+'_'+str(i)+'.svg', transparent=True, bbox_inches='tight')  
    
    # plot unscaled spectral clusters 
    unscaled_spectra_list = []
    for i in range(num_clusters):
        unspectra_cluster_list = []
        # set plotting parameters
        fig = plt.figure(figsize=small_fig_size)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
        ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
        ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
        ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
        # Set the axis limits
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(0.05, 10)
        ax.set_ylim(0.000001, 15)
        # Edit the major and minor tick locations
        x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
        ax.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        ax.xaxis.set_minor_locator(x_minor)
        y_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
        ax.yaxis.set_major_locator(y_major)
        y_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        ax.yaxis.set_minor_locator(y_minor)
        
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        # plot
        rainbow = cm.get_cmap('rainbow', 5)
        line_color = rainbow(i)
        for j in range(GMNum):
            ids = cluster_ids_x.numpy()[j] 
            if ids == i:
                # plot each cluster's spectra
                GM_ID = int(test_latent_list[1][j])
                # plot each cluster's spectra                    
                unspectra_cluster_list.append(un_Sa_dictionary[GM_ID])
                plt.plot(T,un_Sa_dictionary[GM_ID],c=line_color)
        unscaled_spectra_list.append(unspectra_cluster_list)
        # Add the x and y-axis labels
        ax.set_xlabel('Period [seconds]')
        ax.set_ylabel('Spectral Accel. [g]')
        # Save figure
        plt.savefig('./unscaled_'+file_name+'_'+str(i)+'.svg', transparent=True, bbox_inches='tight')  
        
    # plot average for each unscaled spectral clusters 
    # set plotting parameters
    fig = plt.figure(figsize=small_fig_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    # Set the axis limits
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.05, 10)
    ax.set_ylim(0.000001, 15)
    # Edit the major and minor tick locations
    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
    ax.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator(x_minor)
    y_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
    ax.yaxis.set_major_locator(y_major)
    y_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.yaxis.set_minor_locator(y_minor)
    
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    
    # plot
    line_type = ['-g','--c','-.m',':b','-k']
    for i in range(num_clusters):
        ax.plot(T,average(unscaled_spectra_list[i]),line_type[i],linewidth=plt_line_width,label='Cluster '+str(i+1)+' Avg.')
    # Add legend to plot
    ax.legend(loc='lower left', frameon=False)
    # Add the x and y-axis labels
    ax.set_xlabel('Period [seconds]')
    ax.set_ylabel('Spectral Accel. [g]')
    # Save figure
    plt.savefig('./scaled_means_'+file_name+'_'+str(i)+'.svg', transparent=True, bbox_inches='tight')
    
def wind_plot(spectra,T,file_name):
    fig = plt.figure(figsize=small_fig_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    # Set the axis limits
    # ax.set_xscale('log')
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 50)
    # Edit the major and minor tick locations
    # x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
    # ax.xaxis.set_major_locator(x_major)
    # x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    # ax.xaxis.set_minor_locator(x_minor)
    # ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    # ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    # ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.5))
    # plot
    for ii in range(len(spectra)):
        ax.plot(T,spectra[ii,:], linewidth=plt_line_width)
    # Add the x and y-axis labels
    ax.set_xlabel('Time [10min]')
    ax.set_ylabel('Wind Speed. [m/s]')
    # Save figure
    plt.savefig('./'+file_name+'.svg', transparent=True, bbox_inches='tight')
    
def wind_plot_difference(spectra,T,file_name):
    fig = plt.figure(figsize=small_fig_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.xaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='major', size=tick_mj_sz, width=tick_width, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=tick_mn_sz, width=tick_width, direction='in')
    # # Set the axis limits
    # ax.set_xscale('log')
    # ax.set_xlim(0.05, 10)
    # ax.set_ylim(-2.5, 2.5)
    # # Edit the major and minor tick locations
    # x_major = mpl.ticker.LogLocator(base = 10.0, numticks = 5)
    # ax.xaxis.set_major_locator(x_major)
    # x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    # ax.xaxis.set_minor_locator(x_minor)
    # ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    
    # ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    # ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
    # plot
    for ii in range(len(spectra)):
        ax.plot(T,spectra[ii,:], linewidth=plt_line_width)
    # Add the x and y-axis labels
    ax.set_xlabel('Time [10min]')
    ax.set_ylabel('Wind Speed [m/s]')
    # Save figure
    plt.savefig('./'+file_name+'.svg', transparent=True, bbox_inches='tight')