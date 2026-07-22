S = 'B';
%% Save test dataset inputs into a .csv file
file = ['D:\Mis Documentos D\Modtran LUTGen\Software\LUTGen\LUTs_folder\challenge\scenario',...
    S,'\reference\refInterp.h5'];
hdr = readLUThdr(file);
X = hdr.LUTheader; size(X)
writematrix(X,strrep(file,'.h5','.csv'))

%% Calculate baseline (polyfit) and emulator predictions
S = 'A';
file = ['..\LUTs_folder\challenge\scenario',S,'\train\train2000.h5'];
hdr = readLUThdr(file);
X = hdr.LUTheader;
Y = double(h5read(file,'/LUTdata'));

% Interpolation
Xq = importdata(['..\LUTs_folder\challenge\scenario',S,'\reference\refInterp.csv']);
[~,B] = myinterpn(X,Y,[],'polyfit');
tic
Yq = myinterpn(Xq,'polyfit',B);
toc
%%
h5create(['..\LUTs_folder\challenge\results\baseline_',S,'1.h5'],'/LUTdata',size(Yq),'Datatype','single');
h5write(['..\LUTs_folder\challenge\results\baseline_',S,'1.h5'],'/LUTdata',Yq)
h5writeatt(['..\LUTs_folder\challenge\results\baseline_',S,'1.h5'],'/','runtime',0.458)