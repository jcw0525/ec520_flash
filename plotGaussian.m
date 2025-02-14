%x = linspace(-1,1,100);
%y = linspace(-1,1,100);
width = 10;
[x,y] = meshgrid(-width:width,-width:width);

sigma = 10; 
Z = exp(-1/2*(x.^2+y.^2)/sigma^2);
figure; 
surf(x,y,Z)
xlim([-20,20])
ylim([-20,20])
zlim([0 1])