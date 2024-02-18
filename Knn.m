fileID = fopen('Knn2.txt','r');
formatSpec = '%f';
A = fscanf(fileID,formatSpec);

val = 106;

hold on
%set(gca, 'XScale', 'log', 'YScale', 'line');

%loglog([1 val], [1.493648265624854 1.493648265624854 ])

%x = logspace(0,6,val);
x = linspace(1,val,val);
plot(x,A)
xlabel('Wartość k')
ylabel('Dokładność')
ylim([0 100])