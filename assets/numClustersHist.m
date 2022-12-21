clear;clc;
load('numClusters.txt');
hfig=figure;
histogram(numClusters,'FaceColor','none')
set(gca,'XTick',(4:1:8))
set(gca,'FontSize',8,'FontName','Times New Roman')
xlabel('Number of clusters','FontSize',8,'FontName','Times New Roman')
ylabel('Number of grids','FontSize',8,'FontName','Times New Roman')

% save histogram
figWidth=3.5;
figHeight=3;
set(hfig,'PaperUnits','inches');
set(hfig,'PaperPosition',[0 0 figWidth figHeight]);
figname=('.\Fig23.'); %Fig. 23 in the paper
print(hfig,[figname,'tif'],'-r1200','-dtiff');

meanNumClusters=mean(numClusters);