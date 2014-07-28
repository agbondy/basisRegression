function [Yhat,B] = basisRegression(X,Y,varargin)
    % Yhat = basisRegression(X,Y) implements a multiple linear regression model to predict Y, where the
    % regressors are the outputs of causal filters convolved with the history of the some
    % input X.  The filters reduce the number of parameters and confer smoothness on the prediction.
    % X and Y should have the same number of rows (observations)
    % although X can contain multiple columns (i.e. multiple predictor variables).
    % Also X and Y can be cell arrays, corresponding to data from individual trials.  
    % In that case, a single model will be fit to the entire data set,
    % although the output Yhat will now be a cell array of the same size as
    % the inputs.
    %
    % Yhat = basisRegression(X,Y,...,'basis',basis) defines the basis set, of size
    % # samples x # bases.  By default, these are a set of 10 raised cosines with differet lags and
    % frequencies, of length 100 samples 
    % (NB the first 'nsamples' samples of Yhat cannot be
    % predicted, and are by default set to NaN.)  The default model thus
    % will have #basis x #predictors regressors to fit weights to.
    %
    % Yhat = basisRegression(X,Y,...,'nfolds',nfolds) sets the number of
    % cross-validation folds to nfolds (10 by default)
    %
    % [Yhat,B] = basisRegression(X,Y,...) returns the fitted basis function
    % coefficients of the model
    
    % Adrian Bondy, 2014 (based on older code by Wilson Truccolo)
    
    %% process input args
    global state
    state=struct('verbose',0,'nfolds',10);    
    setBasis;
    j=1;
    while j<=length(varargin)
        if strncmpi(varargin{j},'verbose',2)
            state.verbose=1;
        elseif strncmpi(varargin{j},'basis',2)
            j=j+1;
            state.basis=varargin{j};   
        elseif strncmpi(varargin{j},'nfolds',2)
            j=j+1;
            state.nfolds=varargin{j};                   
        end
        j=j+1;
    end 
    %%
    fitBasisRegression(X,Y);
    %%
    [Yhat,B] = evaluateFit;  
end

function fitBasisRegression(spikes,LFP)
    %% initialize constants & process arguments;
    global state
    state.nTaus=size(state.basis,1);
    nBasis=size(state.basis,2);
    if iscell(spikes)
        state.ntrials=length(spikes);
        nCells=size(spikes{1},2);
        dur=0;
        for t=1:state.ntrials
            dur = dur + size(spikes{t},1);
        end
        outputLength=dur-state.nTaus.*state.ntrials;
    else
        state.ntrials=1;
        [dur,nCells]=size(spikes);
        outputLength=dur-state.nTaus;
    end
    Y=zeros(outputLength,1);
    state.trialid = zeros(outputLength,1);
    cells=1:nCells;
    X=zeros(outputLength,nCells*nBasis); % no harm in initializing this too big
    %% build design matrix
    if state.verbose
        fprintf('Building design matrix.\n');
    end
    m=0;
    if state.ntrials>1
        for t=1:state.ntrials
            trialLength=size(spikes{t},1)-state.nTaus;            
            for j=cells(:)'
                for k=1:nBasis
                    m=m+1;
                    x=filter(state.basis(:,k),1,spikes{t}(:,j));
                    X((t-1)*trialLength+[1:trialLength],m)=x(state.nTaus+1:end,:);
                    Y((t-1)*trialLength+[1:trialLength])=LFP{t}(state.nTaus+1:end,:);   
                    state.trialid((t-1)*trialLength+[1:trialLength]) = t;
                end;
            end 
        end
    else
        for j=cells(:)'
            for k=1:nBasis
                m=m+1;
                x=filter(state.basis(:,k),1,spikes(:,j));
                X(:,m)=x(state.nTaus+1:end,:);
            end;
        end   
        Y=LFP(state.nTaus+1:end);                
    end
    X=[ones(outputLength,1) X]; % add constant term
    %% perform 10-fold regression
    if state.verbose
        fprintf('Performing %d-fold regression on matrix of size %d x %d.\n',state.nfolds,size(X,1),size(X,2));
    end
    tic;
    state.foldsize=(outputLength)./state.nfolds;
    state.foldids = ceil(rand(outputLength,1).*state.nfolds);    
    for fold=1:state.nfolds
        B(:,fold)=regress(Y(state.foldids~=fold),X(state.foldids~=fold,:));      
    end
    if state.verbose
        fprintf('Took %s.\n',timestr(toc));    
    end
    state.B=B;
    if ~state.ntrials
        state.dur=dur;
    end
    state.X=X;
    state.Y=Y;
end

function [Yhat,B] = evaluateFit
    global state
    Y=zeros(length(state.Y),1);
    for fold=1:state.nfolds
        Y(state.foldids==fold)=sum(bsxfun(@times,state.B(:,fold)',state.X(state.foldids==fold,:)),2);
    end
    B=mean(state.B,2);
    if state.ntrials>1
        for t=1:state.ntrials
            Yhat{t}=[NaN(state.nTaus,1);Y(state.trialid==t)];
        end
    else
        Yhat=[NaN(state.nTaus,1);Y];
    end
end

function setBasis
    global state
    state.basis = [1.0000    0.5000         0         0         0         0         0         0         0         0
                    0.8987    0.8018    0.1013         0         0         0         0         0         0         0
                    0.6626    0.9728    0.3374         0         0         0         0         0         0         0
                    0.3944    0.9887    0.6056    0.0113         0         0         0         0         0         0
                    0.1715    0.8770    0.8285    0.1230         0         0         0         0         0         0
                    0.0366    0.6878    0.9634    0.3122         0         0         0         0         0         0
                         0    0.4731    0.9993    0.5269    0.0007         0         0         0         0         0
                         0    0.2754    0.9467    0.7246    0.0533         0         0         0         0         0
                         0    0.1229    0.8283    0.8771    0.1717         0         0         0         0         0
                         0    0.0301    0.6708    0.9699    0.3292         0         0         0         0         0
                         0         0    0.5000    1.0000    0.5000         0         0         0         0         0
                         0         0    0.3372    0.9727    0.6628    0.0273         0         0         0         0
                         0         0    0.1982    0.8987    0.8018    0.1013         0         0         0         0
                         0         0    0.0932    0.7908    0.9068    0.2092         0         0         0         0
                         0         0    0.0272    0.6626    0.9728    0.3374         0         0         0         0
                         0         0    0.0007    0.5269    0.9993    0.4731         0         0         0         0
                         0         0         0    0.3944    0.9887    0.6056    0.0113         0         0         0
                         0         0         0    0.2738    0.9459    0.7262    0.0541         0         0         0
                         0         0         0    0.1715    0.8770    0.8285    0.1230         0         0         0
                         0         0         0    0.0918    0.7887    0.9082    0.2113         0         0         0
                         0         0         0    0.0366    0.6878    0.9634    0.3122         0         0         0
                         0         0         0    0.0066    0.5807    0.9934    0.4193         0         0         0
                         0         0         0         0    0.4731    0.9993    0.5269    0.0007         0         0
                         0         0         0         0    0.3700    0.9828    0.6300    0.0172         0         0
                         0         0         0         0    0.2754    0.9467    0.7246    0.0533         0         0
                         0         0         0         0    0.1923    0.8941    0.8077    0.1059         0         0
                         0         0         0         0    0.1229    0.8283    0.8771    0.1717         0         0
                         0         0         0         0    0.0686    0.7528    0.9314    0.2472         0         0
                         0         0         0         0    0.0301    0.6708    0.9699    0.3292         0         0
                         0         0         0         0    0.0074    0.5856    0.9926    0.4144         0         0
                         0         0         0         0         0    0.5000    1.0000    0.5000         0         0
                         0         0         0         0         0    0.4165    0.9930    0.5835    0.0070         0
                         0         0         0         0         0    0.3372    0.9727    0.6628    0.0273         0
                         0         0         0         0         0    0.2639    0.9408    0.7361    0.0592         0
                         0         0         0         0         0    0.1982    0.8987    0.8018    0.1013         0
                         0         0         0         0         0    0.1411    0.8481    0.8589    0.1519         0
                         0         0         0         0         0    0.0932    0.7908    0.9068    0.2092         0
                         0         0         0         0         0    0.0552    0.7284    0.9448    0.2716         0
                         0         0         0         0         0    0.0272    0.6626    0.9728    0.3374         0
                         0         0         0         0         0    0.0091    0.5950    0.9909    0.4050         0
                         0         0         0         0         0    0.0007    0.5269    0.9993    0.4731         0
                         0         0         0         0         0         0    0.4596    0.9984    0.5404    0.0016
                         0         0         0         0         0         0    0.3944    0.9887    0.6056    0.0113
                         0         0         0         0         0         0    0.3321    0.9710    0.6679    0.0290
                         0         0         0         0         0         0    0.2738    0.9459    0.7262    0.0541
                         0         0         0         0         0         0    0.2201    0.9143    0.7799    0.0857
                         0         0         0         0         0         0    0.1715    0.8770    0.8285    0.1230
                         0         0         0         0         0         0    0.1287    0.8348    0.8713    0.1652
                         0         0         0         0         0         0    0.0918    0.7887    0.9082    0.2113
                         0         0         0         0         0         0    0.0610    0.7394    0.9390    0.2606
                         0         0         0         0         0         0    0.0366    0.6878    0.9634    0.3122
                         0         0         0         0         0         0    0.0185    0.6346    0.9815    0.3654
                         0         0         0         0         0         0    0.0066    0.5807    0.9934    0.4193
                         0         0         0         0         0         0    0.0007    0.5266    0.9993    0.4734
                         0         0         0         0         0         0         0    0.4731    0.9993    0.5269
                         0         0         0         0         0         0         0    0.4208    0.9937    0.5792
                         0         0         0         0         0         0         0    0.3700    0.9828    0.6300
                         0         0         0         0         0         0         0    0.3214    0.9670    0.6786
                         0         0         0         0         0         0         0    0.2754    0.9467    0.7246
                         0         0         0         0         0         0         0    0.2322    0.9223    0.7678
                         0         0         0         0         0         0         0    0.1923    0.8941    0.8077
                         0         0         0         0         0         0         0    0.1558    0.8626    0.8442
                         0         0         0         0         0         0         0    0.1229    0.8283    0.8771
                         0         0         0         0         0         0         0    0.0938    0.7915    0.9062
                         0         0         0         0         0         0         0    0.0686    0.7528    0.9314
                         0         0         0         0         0         0         0    0.0474    0.7124    0.9526
                         0         0         0         0         0         0         0    0.0301    0.6708    0.9699
                         0         0         0         0         0         0         0    0.0168    0.6285    0.9832
                         0         0         0         0         0         0         0    0.0074    0.5856    0.9926
                         0         0         0         0         0         0         0    0.0018    0.5427    0.9982
                         0         0         0         0         0         0         0         0    0.5000    1.0000
                         0         0         0         0         0         0         0         0    0.4578    0.9982
                         0         0         0         0         0         0         0         0    0.4165    0.9930
                         0         0         0         0         0         0         0         0    0.3762    0.9844
                         0         0         0         0         0         0         0         0    0.3372    0.9727
                         0         0         0         0         0         0         0         0    0.2997    0.9581
                         0         0         0         0         0         0         0         0    0.2639    0.9408
                         0         0         0         0         0         0         0         0    0.2301    0.9209
                         0         0         0         0         0         0         0         0    0.1982    0.8987
                         0         0         0         0         0         0         0         0    0.1685    0.8743
                         0         0         0         0         0         0         0         0    0.1411    0.8481
                         0         0         0         0         0         0         0         0    0.1159    0.8202
                         0         0         0         0         0         0         0         0    0.0932    0.7908
                         0         0         0         0         0         0         0         0    0.0730    0.7601
                         0         0         0         0         0         0         0         0    0.0552    0.7284
                         0         0         0         0         0         0         0         0    0.0400    0.6958
                         0         0         0         0         0         0         0         0    0.0272    0.6626
                         0         0         0         0         0         0         0         0    0.0169    0.6289
                         0         0         0         0         0         0         0         0    0.0091    0.5950
                         0         0         0         0         0         0         0         0    0.0037    0.5609
                         0         0         0         0         0         0         0         0    0.0007    0.5269
                         0         0         0         0         0         0         0         0         0    0.4930
                         0         0         0         0         0         0         0         0         0    0.4596
                         0         0         0         0         0         0         0         0         0    0.4267
                         0         0         0         0         0         0         0         0         0    0.3944
                         0         0         0         0         0         0         0         0         0    0.3628
                         0         0         0         0         0         0         0         0         0    0.3321
                         0         0         0         0         0         0         0         0         0    0.3024
                         0         0         0         0         0         0         0         0         0    0.2738
                         0         0         0         0         0         0         0         0         0    0.2463];

end