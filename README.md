# Hurricane Wind Records Clustering and Selection
To select representative records for incremental dynamic analysis (IDA), the hurricane wind records are clustered by their patterns in duration and changing of wind speed and directions, and then a set of wind records can be selected from each cluster. The number of wind records selected from each cluster should be propotional to the total number of wind records in that cluster, which can preserve the relative propotion of different patterns.  

Since the wind records are time series of 2D vectors with different length, it is difficult to cluster the records directly. To facititate the clustering process, the wind records are first compressed using an artificial neural network autoencoder to obtain low dimensional latent features of the time series. The architecture of the autoencoder is presented in the following figure. It shows that the 2D wind speed records are compressed into 5 latent features through the encoder process , which then are expanded to the reconstructed 2D wind records through the decoder process. The training of this autoencder is conducted by minimizing the error between the reconstructed wind records and the original wind records, which ensures that the 5 latent features can represent the important patterns of the wind records.  
![Alt text](/assets/Figure1.jpg)  
After the training process, all 2D wind speed time series are converted into latent feature vectors, on which the k-means algorithm is applied for clustering. The wind records are divided into 4 clusters. The following figure shows the first 3 latent features for the 4 clusters, from which it may be seen that the hurricanes are clustered well because different clusters have almost no overlap and the latent features for each cluster are gathered closely around their centroid.  
![Alt text](/assets/Figure2.jpg)  
The following two figures illustrate the hurricane wind speeds of Cluster 1.  
![Alt text](/assets/Figure3.jpg)
![Alt text](/assets/Figure4.jpg)  
The following two figures illustrate the hurricane wind speeds of Cluster 2.  
![Alt text](/assets/Figure5.jpg)
![Alt text](/assets/Figure6.jpg)  
It is seen that the clustering results are reasonable because hurricane wind speeds and durations within each cluster have similar patterns. The other two clusters provide similar results to these.