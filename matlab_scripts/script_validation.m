
%% Challenge evaluation script

% Path to results of participants
path = ['D:\Mis Documentos D\Trabajo\UVEG\AI4ART postdoc 2022-2024\Projects\ELIAS\EmulationChallenge\',...
    'rtm_emulation\results\']; % <-- adapt for use in ISP server
path = 'D:\Mis Documentos D\Modtran LUTGen\Software\LUTGen\LUTs_folder\challenge\results\'

% Check how many .h5 files we have (i.e. how many contributions from participants)
d = dir([path,'*.h5']);

% Find how many different models we have
pattern = '^(.*)_[AB][1-3]\.h5$';
tokens = regexp({d.name}, pattern, 'tokens');
% Extract model names from tokens
model_names = cellfun(@(x) x{1}{1}, tokens(~cellfun('isempty', tokens)), 'UniformOutput', false);
model_names = unique(model_names); % unique model names
n = numel(model_names);


% Scenario-track IDs
% Note! edit configurations depending on the reference data available
S = {'A','B'}; tracks = [1 2];
trackNames = {'refInterp','refExtrap','refReal'};
m = numel(S)*numel(tracks); % number of scenario-track combinations (hard-coded value)

% Memory allocation for mean relative error (MRE) and ranking
mre = nan(n,m); rnk = nan(n,m);

%% Start evaluation of error metrics
for s = 1:numel(S) % for each scenario    
    for t = 1:numel(tracks) % for each track
        j = t + (s-1)*numel(tracks) % scenario-track index
        sprintf('Scenario %s - track: %s',S{s},trackNames{t})
        
        % Read reference data:
        file = strrep(path,'results\',...
            sprintf('scenario%s\\reference\\%s.h5',S{s},trackNames{t}));
        L = double(h5read(file,'/LUTdata')); % read TOA radiance (mW/m2/sr/nm)
        sza = h5read(file,'/LUTheader'); 
        if s==1, sza = sza(7,:);
        else, sza = sza(5,:);
        end
        wvl = h5read(file,'/wvl');
        n_wvl = numel(wvl); % number of wavelengths
        
        % Read reference surface reflectance
        file = strrep(path,'results\','scenarioA\\reference\\refldb.txt');
        rho_ref = importdata(file);
        rho_ref = spline(rho_ref(:,1),rho_ref(:,2),wvl);
        
        % Evaluate each model on the current scenario-track
        for i = 1:n
            file = sprintf('%s_%s%d.h5',model_names{i},S{s},t); % filename
            sprintf('Model %d (%s)',i,file)
            if exist([path,file]) % only if file exists
                yq = double(h5read([path,file],'/LUTdata')); % read model predictions
                               
                if s==1 % scenario A
                    % Retrieve surface reflectance:
                    rho_ret = retrieve_reflectance(L,yq,sza,n_wvl);
                    % Evaluate error metric:
                    mre(i,j) = error_metric_A(rho_ref,rho_ret,wvl);
                else % scenario B
                    % Calculate TOA radiance:
                    Ltoa_ret = toa_radiance(rho_ref,yq,sza,n_wvl);
                    mre(i,j) = error_metric_B(L,Ltoa_ret);
                end
            end
        end
        % Calculate inidividual ranking:
        [~,rnk_aux] = sort(mre(:,j));
        [~,rnk(:,j)] = sort(rnk_aux);
        % Assign worst rank to missing submissions:
        rnk(isnan(mre(:,j)),j) = n;
    end
end
% Calculate average ranking
[rnk_avg,finalScore] = compute_final_ranks(rnk);

table(model_names',finalScore,rnk_avg,mre)
%% Support functions
function rho = retrieve_reflectance(Ltoa,Y,sza,n)
    % Split into transfer functions:
    L0 = Y(1:n,:);
    E  = Y(1+n:2*n,:).*cosd(sza) + Y(1+2*n:3*n,:);
    Sa = Y(1+3*n:4*n,:);
    T  = Y(1+4*n:5*n,:) + Y(1+5*n:6*n,:);

    % Perform atmospheric correction and retrieve surface reflectance:
    rho = pi*(Ltoa - L0)./(E.*T + pi*(Ltoa - L0).*Sa);
end

function Ltoa = toa_radiance(rho,Y,sza,n)
    % Split into transfer functions:
    L0 = Y(1:n,:);
    E  = Y(1+n:2*n,:).*cosd(sza) + Y(1+2*n:3*n,:);
    Sa = Y(1+3*n:4*n,:);
    T  = Y(1+4*n:5*n,:) + Y(1+5*n:6*n,:);

    % Calculate TOA radiance
    Ltoa = L0 + (1/pi)*E.*T.*rho./(1-Sa.*rho);
end
        
function mre = error_metric_A(rho_ref,rho_ret,wvl)
    % Relative error:
    re = 100*abs(rho_ret-rho_ref)./rho_ref;
    % Calculate mean for all samples:
    re = mean(re,2,'omitnan');
    % Calculate spectral average avoiding absorption bands:
    idx = ~((wvl > 931 & wvl < 945) |...
            (wvl > 1100 & wvl < 1160) |...
            (wvl > 1300 & wvl < 1500) |...
            (wvl > 1750 & wvl < 1980) | (wvl > 2420));
    mre = mean(re(idx),'omitnan');
end

function mre = error_metric_B(Ltoa_ref,Ltoa_ret)
    % Relative error:
    re = 100*abs(Ltoa_ref-Ltoa_ret)./Ltoa_ref;
    % Calculate mean for all samples:
    re = mean(re,2,'omitnan');
    % Calculate spectral average:
    mre = mean(re,'omitnan');
end

function [rnk_avg, final_ranks] = compute_final_ranks(rnk)
% COMPUTE_FINAL_RANKS computes the weighted average ranking and
% the standard competition ranks for a given ranking matrix.
%
% Inputs:
%   rnk - an (n x 4) matrix of individual ranks for each participant (n models)
%         Columns are assumed to be in the order:
%         [AC-Interp, AC-Extra, CO2-Interp, CO2-Extra]
%
% Outputs:
%   rnk_avg_weighted - weighted average rank for each participant
%   final_ranks       - standard competition rank based on the average

    n = size(rnk, 1); % number of participants

    % Define weights (Interpolation: 0.65, Extrapolation: 0.35)
    % Each scenario (AC, CO2) contributes equally
    weights = [0.325, 0.175, 0.325, 0.175]; % corresponds to [AC-I, AC-E, CO2-I, CO2-E]

    % Compute weighted average ranking
    rnk_avg = rnk*weights';

    % Compute standard competition ranking
    [sorted_scores, idx_sorted] = sort(rnk_avg);
    final_ranks = zeros(n,1);

    i = 1;
    while i <= n
        % Find all tied scores at this rank
        tie_value = sorted_scores(i);
        tied = find(abs(sorted_scores - tie_value) < 1e-8);
        tied = tied(tied >= i); % consider only from current index onward
        k = numel(tied);

        % Assign same rank to all tied participants
        final_ranks(idx_sorted(tied)) = i;

        i = i + k;
    end
end
