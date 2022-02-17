clear;clc;
load('numClusters.txt');
histogram(numClusters)
set(gca,'XTick',(4:1:8))
xlabel('Number of clusters')
ylabel('Number of grids')