fileID = fopen('ML_hidden.txt','r');
formatSpec = '%f';
A = fscanf(fileID,formatSpec);
B = reshape(A,[10,10]);

val = 10;

hold on
%set(gca, 'XScale', 'log', 'YScale', 'line');

%loglog([1 val], [1.493648265624854 1.493648265624854 ])

%x = logspace(0,6,val);
%x = linspace(1,val,val);
[X,Y] = meshgrid(1:1:10,1:1:10);
%x = [0, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1, 5, 10, 50, 100];
surf(X,Y,B)
%%xlabel('Wartość max_iter')
%%ylabel('Dobroć')
%%ylim([0 100])
