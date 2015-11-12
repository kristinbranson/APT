function [phisPr,phisPrAll,phisPrTAll]=test(expdirs,trainresfile,testresfile,varargin)

[moviefilestr,traindeps,restart,firstframe,endframe,trxfilestr,flies,winrad,readframe,nframes] = ...
  myparse(varargin,...
  'moviefilestr','movie_comb.avi','traindeps',[],'restart',[],...
  'firstframe',1,'endframe',inf,'trxfilestr','','flies',[],'winrad',[],...
  'readframe',[],'nframes',[]);

nperiter = 100;

if ~iscell(expdirs),
  expdirs = {expdirs};
end

% if ~iscell(trainresfiles),
%   trainresfiles = {trainresfiles};
% end

tmp = who('-file',trainresfile);
fns = intersect({'regPrm','prunePrm','regModel','H0','doscale'},tmp);

load(trainresfile,fns{:});
if ~iscell(regPrm), %#ok<NODEF>
  regPrm = {regPrm};
end
if ~iscell(prunePrm), %#ok<NODEF>
  prunePrm = {prunePrm};
end
if ~iscell(regModel), %#ok<NODEF>
  regModel = {regModel};
end

if ismember('traindeps',tmp),
  load(trainresfile,'traindeps');
else
  traindeps = 0;
end
if ~exist('doscale','var'),
  doscale = false;
end

nregressors = numel(regPrm);

if isempty(traindeps),
  traindeps = [0,ones(1,nregressors-1)];
end

assert(numel(traindeps) == nregressors);

istrx = ~isempty(trxfilestr) && numel(flies) == numel(expdirs) && ~isempty(winrad);

isrestart = false;
if ~isempty(restart),
  phisPr = restart.phisPr;
  phisPrAll = restart.phisPrAll;
  phisPrTAll = restart.phisPrTAll;
  if size(phisPr,2) == numel(expdirs) && size(phisPrAll,2) == numel(expdirs),
    if size(phisPr,1) < nregressors,
      phisPr = [phisPr;cell(nregressors-size(phisPr,1),numel(expdirs))];
    end
    if size(phisPrAll,1) < nregressors,
      phisPrAll = [phisPrAll;cell(nregressors-size(phisPrAll,1),numel(expdirs))];
    end
    regisdone = all(cellfun(@(x) ~isempty(x) && all(~isnan(x(:))), phisPr) & ...
      cellfun(@(x) ~isempty(x) && all(~isnan(x(:))), phisPrAll),2);
    registart = find(~regisdone,1);
    if isempty(registart),
      return;
    end
    isrestart = true;
  end
end
if ~isrestart,
  phisPr = cell(nregressors,numel(expdirs));
  phisPrAll = cell(nregressors,numel(expdirs));
  phisPrTAll = cell(nregressors,numel(expdirs));
  registart = 1;
end

if isdeployed && ischar(firstframe),
  firstframe = str2double(firstframe);
end
if isdeployed && ischar(endframe),
  endframe = str2double(endframe);
end

for regi = registart:nregressors,

  for expi = 1:numel(expdirs),

    fprintf('Regressor %d/%d, experiment %d/%d\n',regi,nregressors,expi,numel(expdirs));

    expdir = expdirs{expi};
    
    if isempty(readframe) || isempty(nframes),
      if isempty(moviefilestr),
        [readframe,nframes,fid] = get_readframe_fcn(expdir);
      else
        [readframe,nframes,fid] = get_readframe_fcn(fullfile(expdir,moviefilestr));
      end
    else
      fid = [];
    end
    
    endframecurr = min(nframes,endframe);
    N = endframecurr - firstframe + 1;
    
    if istrx,  
      trx = load_tracks(fullfile(expdir,trxfilestr));
      fly = flies(expi);
    end


    phisPr{regi,expi} = nan([N,regPrm{regi}.model.D]);
    phisPrAll{regi,expi} = nan([N,regPrm{regi}.model.D,prunePrm{regi}.numInit]);
    phisPrTAll{regi,expi} = nan([N,prunePrm{regi}.numInit,regPrm{regi}.model.D,regModel{regi}.T+1]);
    
    if regi > 1 && isfield(prunePrm{regi},'initfcn')
      bboxesTeAll = prunePrm{regi}.initfcn(phisPr{traindeps(regi),expi});
    end
    
    for t_i=firstframe:nperiter:endframecurr,
      t_f=min(t_i+nperiter-1,endframecurr);
      off_i = t_i - firstframe + 1;
      off_f = t_f - firstframe + 1;
      nframescurr = t_f-t_i+1;
      fprintf('\nTracking: frames %i-%i\n',t_i,t_f);
      IsTe = cell(nframescurr,1);
      xcurr = nan(nframescurr,1);
      ycurr = nan(nframescurr,1);
      for k=1:nframescurr,
        t=t_i+k-1;
        im = readframe(t);
        imsz = size(im);
        %im = imrotate(im,180);
        if size(im,3) > 1,
          im = rgb2gray(im);
        end
        if istrx,
          toff = t-trx(fly).firstframe+1;
          xcurr(k) = round(trx(fly).x(toff));
          ycurr(k) = round(trx(fly).y(toff));
          im = padgrab(im,255,ycurr(k)-winrad,ycurr(k)+winrad,xcurr(k)-winrad,xcurr(k)+winrad);
        end
        IsTe{k} = im;
      end
      imsz = size(IsTe{1});
      if exist('H0','var'),
        bigim = cat(1,IsTe{1:nframescurr});
        bigimnorm = histeq(bigim,H0);
        IsTe(1:nframescurr) = mat2cell(bigimnorm,repmat(imsz(1),[1,nframescurr]),imsz(2));
      end
      if doscale,
        maxv = 0;
        for tmpi = 1:nframescurr,
          maxv = max(maxv,max(IsTe{tmpi}(:)));
        end
        maxv = single(maxv);
        for tmpi = 1:nframescurr,
          IsTe{tmpi} = uint8(single(IsTe{tmpi})/maxv*255);
        end
      end
      
      if regi > 1 && isfield(prunePrm{regi},'initfcn')
        bboxesTe = bboxesTeAll(off_i:off_f,:);
      else
        fprintf('Using default bboxes!\n');
        bboxesTe = repmat([1,1,imsz([2,1])],[numel(IsTe),1]);
      end
      
      if isfield(prunePrm{regi},'initlocs'),
        piT = repmat(permute(prunePrm{regi}.initlocs,[3,1,2]),[nframescurr,1,1]);
      else
        piT = [];
      end

      [phisPr{regi,expi}(off_i:off_f,:),phisPrAll{regi,expi}(off_i:off_f,:,:),~,~,p_t] = test_rcpr([],bboxesTe,IsTe,regModel{regi},regPrm{regi},prunePrm{regi},piT);
      
      sz = size(p_t);
      p_t = reshape(p_t,[nframescurr,prunePrm{regi}.numInit,sz(end-1:end)]);
      
      
      if istrx,
        xoff = xcurr - winrad - 1;
        yoff = ycurr - winrad - 1;
        nfids = regModel{regi}.model.nfids;
        phisPr{regi,expi}(off_i:off_f,1:nfids) = ...
          bsxfun(@plus,phisPr{regi,expi}(off_i:off_f,1:nfids),xoff);
        phisPr{regi,expi}(off_i:off_f,nfids+1:end) = ...
          bsxfun(@plus,phisPr{regi,expi}(off_i:off_f,nfids+1:end),yoff);
        phisPrAll{regi,expi}(off_i:off_f,1:nfids,:) = ...
          bsxfun(@plus,phisPrAll{regi,expi}(off_i:off_f,1:nfids,:),xoff);
        phisPrAll{regi,expi}(off_i:off_f,nfids+1:end,:) = ...
          bsxfun(@plus,phisPrAll{regi,expi}(off_i:off_f,nfids+1:end,:),yoff);
        p_t(:,:,1:nfids,:) = bsxfun(@plus,p_t(:,:,1:nfids,:),xoff);
        p_t(:,:,nfids+1:end,:) = bsxfun(@plus,p_t(:,:,nfids+1:end,:),yoff);
      end
      
      phisPrTAll{regi,expi}(off_i:off_f,:,:,:) = p_t;
      
    end
    
    K = prunePrm{regi}.numInit;
    appearancecost = nan(N,K);
    pRT1 = reshape(phisPrAll{regi,expi},[N,regPrm{regi}.model.nfids,regPrm{regi}.model.d,K]);
    for n = 1:N,
      pr = 0;
      for part = 1:regPrm{regi}.model.nfids,
        d = pdist(reshape(pRT1(n,part,:,:),[regPrm{regi}.model.d,K])').^2;
        %d = pdist(reshape(pRT(n,[1,3],:),[2,RT1])').^2;
        w = sum(squareform(exp( -d/prunePrm{regi}.maxdensity_sigma^2/2 )),1);
        w = w / sum(w);
        pr = pr - log(w);
      end
      appearancecost(n,:) = pr;
    end
      
    % convert to 3d from multiple views?
    if isfield(prunePrm{regi},'motion_2dto3D') && prunePrm{regi}.motion_2dto3D,
      
      nparts3D = floor(regPrm{regi}.model.nfids/2);
      % xL is N x nparts3D x 2 x RT1
      xL = pRT1(:,1:nparts3D,:,:);
      xR = pRT1(:,nparts3D+1:2*nparts3D,:,:);
      xL = permute(xL,[3,1,2,4]);
      xR = permute(xR,[3,1,2,4]);
      % now xL is 2 x N x nparts3D x RT1
      
      % X is 3 x N x nparts3D x RT1
      X = stereo_triangulation(xL,xR,prunePrm{regi}.calibrationdata.om0,...
        prunePrm{regi}.calibrationdata.T0,prunePrm{regi}.calibrationdata.fc_left,...
        prunePrm{regi}.calibrationdata.cc_left,prunePrm{regi}.calibrationdata.kc_left,...
        prunePrm{regi}.calibrationdata.alpha_c_left,prunePrm{regi}.calibrationdata.fc_right,...
        prunePrm{regi}.calibrationdata.cc_right,prunePrm{regi}.calibrationdata.kc_right,...
        prunePrm{regi}.calibrationdata.alpha_c_right);
      X = reshape(X,[3,N,nparts3D,K]);
        
      X = reshape(permute(X,[1,3,2,4]),[3*nparts3D,N,K]);
      % X is D x N x RT1
      
    else
      % phisPrAll{regi,expi} is N x D x K
      % want X to be D x N x K
      X = permute(phisPrAll{regi,expi},[2,1,3]);
      
    end
      
    % Xbest is D x N
    [Xbest,idxbest,totalcost,poslambda] = ChooseBestTrajectory(X,appearancecost,prunePrm{regi}.motionparams{:}); %#ok<NASGU,ASGLU>
    
    if prunePrm{regi}.motion_2dto3D,
      % xLbest is initially 2 x N x nparts3D
      xLbest = nan([regPrm{regi}.model.d,N,nparts3D]);
      xRbest = nan([regPrm{regi}.model.d,N,nparts3D]);
      for i = 1:N,
        xLbest(:,i,:) = xL(:,i,:,idxbest(i));
        xRbest(:,i,:) = xR(:,i,:,idxbest(i));
      end
      % permute it to be N x nparts3D x 2
      xLbest = permute(xLbest,[2,3,1]);
      xRbest = permute(xRbest,[2,3,1]);
      % N x nfids x 2
      xbest = cat(2,xLbest,xRbest);
      phisPr{regi,expi} = reshape(xbest,[N,regPrm{regi}.model.nfids*regPrm{regi}.model.d]);
      
    else
      phisPr{regi,expi} = permute(Xbest,[2,1]);
    end
    
    
    for fidi = 1:numel(fid),
      if fid(fidi) > 1,
        fclose(fid(fidi));
      end
    end
    
  end
  
end


if ~isempty(testresfile),
  
  save(testresfile,'phisPr','phisPrAll','expdirs','moviefilestr','trainresfile');
  
end
