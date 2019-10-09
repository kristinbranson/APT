classdef SelectFeatures
  % Select Features usually by correlation etc  
  
  methods (Static)
    
    function [stdFtrs,dfFtrs] = statsFtrs(X,ftrPrm)
      % Compute feature stats to be used by fast correlation selection
      %
      % - If ftrPrm.metatype=='single', then stdFtrs and dfFtrs are column
      % SDs and means of X, resp.
      % 
      % - If ftrPrm.metatype=='diff', then dfFtrs is as before but stdFtrs 
      % is actually FxF upper triangular, std-of-differences, ie 
      % stdFtrs(i,j) is std(X(:,i)-X(:,j)). See stdFtrs1.

      N = size(X,1);
      
      if isfield(ftrPrm,'stdsamples') && ~isempty(ftrPrm.stdsamples),
        nsample = ftrPrm.stdsamples(2)-ftrPrm.stdsamples(1)+1;
      else
        assert(isfield(ftrPrm,'nsample_std'));
        nsample = ftrPrm.nsample_std;
      end
      
      muFtrs = mean(X);
      dfFtrs = bsxfun(@minus,X,muFtrs);

      switch ftrPrm.metatype
        case 'single'
          stdFtrs = std(X); 
        case 'diff'          
          if nsample < N
            if isfield(ftrPrm,'stdsamples') && ~isempty(ftrPrm.stdsamples),
              dosample = ftrPrm.stdsamples(1):ftrPrm.stdsamples(2);
            else
              dosample = rand(N,1) <= nsample/N;
            end
          else
            dosample = true(N,1);
          end
          stdFtrs = stdFtrs1(X(dosample,:));
        otherwise
          assert(false);
      end
    end

    function use = testSelectSingles
      N = 2e3;
      F = 1600;
      S = 5;
      X = rand(N,F);
      B = rand(N,S);
      
      tic; use1 = SelectFeatures.selectSingle_slow(X,B); toc
      tic; use2 = SelectFeatures.selectSingleWrap(X,B); toc
      tic; use3 = SelectFeatures.selectFeatSingleWrap(X,B); toc
      assert(isequal(use1(:),use2(:),use3(:)));
      
      use = use1(:);
    end    
    
  end
  
  methods (Static)
    
    function use = selectSingle_slow(X,B)
      % Naive impl
      %
      % X: [NxF] features
      % B: [NxS] projections of yTar onto S directions
      %
      % use: [S] selected features; indices into cols of X
      
      [N,F] = size(X);
      [N2,S] = size(B);
      assert(N==N2);
      
      use = nan(S,1);
      tfused = false(F,1);
      for iS = 1:S
        vals = nan(F,1);
        for iF = 1:F
          tmp = corrcoef(X(:,iF),B(:,iS));
          vals(iF) = abs(tmp(1,2));
        end
        [~,idx] = sort(vals,1,'descend');
        for iF = idx(:)'
          if ~tfused(iF)
            use(iS) = iF;
            tfused(iF) = true;
            break;
          end
        end
      end
    end
    
    function use = selectSingle(X,Xmu,Xsd,B,Bmu,Bsd)
      %
      % X: [NxF] features
      % Xmu: [1xF] column means of X
      % Xsd: [1xF] column stds of X
      % B: [NxS] projections of target shapes onto S directions
      % Bmu: [1xS] column means of b
      % Bsd: [1xS] column stds of b
      %
      % use: [S] selected features; indices into columns of X      
      
      [N,F] = size(X);
      assert(isequal(size(Xmu),[1 F]));
      assert(isequal(size(Xsd),[1 F]));
      [N2,S] = size(B);
      assert(N==N2);
      assert(isequal(size(Bmu),[1 S]));
      assert(isequal(size(Bsd),[1 S]));
      
      dX = bsxfun(@minus,X,Xmu);
      dB = bsxfun(@minus,B,Bmu);
      
      use = nan(1,S);
      tfused = false(F,1); % flag which features have been accounted for
      for iS = 1:S
        bestValSeen = -inf;
        bestValIF = nan;
        for iF = 1:F
          if ~tfused(iF)
            val = abs(sum(dX(:,iF).*dB(:,iS))) / Xsd(iF) / Bsd(iS);
            if val>bestValSeen
              bestValSeen = val;
              bestValIF = iF;
            end
          end
        end
        
        use(iS) = bestValIF;
        assert(~tfused(bestValIF));
        tfused(bestValIF) = true;
      end
      
    end
      
    function use = selectSingleWrap(X,B)
      use = SelectFeatures.selectSingle(X,mean(X),std(X),B,mean(B),std(B));      
    end
    
    function use = selectFeatSingleWrap(X,B)
      dX = bsxfun(@minus,X,mean(X,1));
      Xsd = std(X,[],1);
      dB = bsxfun(@minus,B,mean(B,1));
      Bsd = std(B,[],1);
      use = selectFeatSingle(dX,Xsd,dB,Bsd);
    end
    
  end
  
%   %%
% N = 6200;
% D = 14;
% F = 400;
% S = 5;
% type = 2;
% %%
% pTar = rand(N,D)*10;
% ftrs = rand(N,F)*2-1;
% muFtrs = mean(ftrs);
% dfFtrs = bsxfun(@minus,ftrs,muFtrs);
% stdFtrs = stdFtrs1(ftrs);    
% b = rand(D,S)*2-1; 
% b = bsxfun(@rdivide,b,sqrt(sum(b.^2,1)));
% scalar = pTar*b; 
% stdSc = std(scalar); 
% muSc = mean(scalar);
% %%
% tic;
% use = selectCorrFeat1(pTar,ftrs,type,stdFtrs,dfFtrs,scalar,stdSc,muSc);
% toc
% %%
% tic;
% [use1,maxco1] = selectCorrFeat1_AL(pTar,ftrs,type,stdFtrs,dfFtrs,scalar,stdSc,muSc,false);
% toc
% %%
% %%
% tic;
% [use2,maxco2] = selectCorrFeat1_AL(pTar,ftrs,type,stdFtrs,dfFtrs,scalar,stdSc,muSc,true);
% toc
% 
% 
%   
end