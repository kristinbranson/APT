function [X3d,x2d,trpeconf,isspecial,prefview,...
  x2dcmp,... % nan(nfrm,3,2,nview,npts); % 2nd dim: [best/chosen, trkorig, trkorigRP]
  hmapscore,hmapscorecmp,hmaproi ... % these outputs only relevant when hmbig supplied
  ] = viewpref3drecon(trk1,trk2,crobj,varargin)

% very experimental for RF!

% TODO: preferviewnborrad

[ipts,frms,dxyz,mdnhmsigma2,unethmsarereconned,rpethresh,...
  doprefview,preferviewunetconf,preferviewlambda,preferviewnborrad,...
  hmbig,roisEPline,...
  wbObj] = myparse(varargin,...
  'ipts',[],... % absolute point indices
  'frms',[],... % absolute frame numbers
  'dxyz',0.01,...
  'mdnhmsigma2',3.2^2,... % var for mdn fake hmap
  'unethmsarereconned',true,... % if false, we massage/flatten mdn hmaps to make them comparable to 'real' unet hmaps, which do not shrink to -inf in logspace
  'rpethresh',6,... % if rpemn><this>, then do the fancy 3dgridRP. specify as inf to never do it
  'preferview',true,... % if true, when doing the fancy, if one view has more conf then prefer it
  'preferviewunetconf',0.7,...
  'preferviewlambda',3,... % used when preferview==true
  'preferviewnborrad',2,... % used when preferview==true. in 'weak' view, include nboring frames with this rad. use 0 for no-nbors
  'hmbig',[],...
  'roisEPline',[],... % [2x4], used if hmbig not supplied. rois(iview,:) is [xlo xhi ylo yhi] for view iview for epipolar line computation
  'wbObj',WaitBarWithCancelCmdline('3dpp')  ...
);

if isempty(ipts)  
  ipts = 1:size(trk1.pTrk,1);
end
if isempty(frms)
  frms = trk1.pTrkFrm;
end

tfHM = ~isempty(hmbig);
if ~tfHM
  assert(~isempty(roisEPline));
end

assert(isequal(size(trk1.pTrk),size(trk2.pTrk)));
assert(isequal(1,size(trk1.pTrk,4),size(trk2.pTrk,4)),...
  'Expected single-target trkfiles');
assert(isequal(trk1.pTrkFrm,trk2.pTrkFrm));
[tf,ifrmTrk] = ismember(frms,trk1.pTrkFrm);
assert(all(tf));
  
npts = numel(ipts);
nfrm = numel(frms);
nview = 2;

if tfHM
  tfhmidx = ismember(double(hmbig.frmagg),frms);
  assert(isequal(double(hmbig.frmagg(tfhmidx)),frms(:)));
  hmbigrawfloat = hmbig.hmagg(tfhmidx,:,:,:,:);
  [~,hmnr,hmnc,~,~] = size(hmbigrawfloat);
  szassert(hmbigrawfloat,[nfrm hmnr hmnc nview npts]); % XXX assumes hmbigarr already selected for ipts
end

X3d = nan(nfrm,3,npts);
x2d = nan(nfrm,2,nview,npts);
trpeconf = cell(npts,1); % trpeconf{ipt} is [nfrm] table
isspecial = false(nfrm,npts);
prefview = zeros(nfrm,npts); % 0 = no pref, etc
x2dcmp = nan(nfrm,3,2,nview,npts); % x-y for alscorecmp
hmapscore = nan(nfrm,npts);
hmapscorecmp = nan(nfrm,3,npts); % best3d, orig, origRPs
hmaproi = nan(nfrm,4,nview,npts);
%origTrkStuff = cell(npts,1);

wbObj.startPeriod('Points','shownumden',true,'denominator',npts);

for iipt=1:npts
  wbObj.updateFracWithNumDen(iipt);

  ipt = ipts(iipt);
    
  ptrk1 = reshape(trk1.pTrk(ipt,:,ifrmTrk),[2 nfrm]);
  ptrk2 = reshape(trk2.pTrk(ipt,:,ifrmTrk),[2 nfrm]);
  ptrk1u = reshape(trk1.pTrklocs_unet(ipt,:,ifrmTrk),[2 nfrm]);
  ptrk2u = reshape(trk2.pTrklocs_unet(ipt,:,ifrmTrk),[2 nfrm]);
  conf1 = trk1.pTrkconf(ipt,ifrmTrk); % 1xnfrm
  conf1u = trk1.pTrkconf_unet(ipt,ifrmTrk); % etc
  conf2 = trk2.pTrkconf(ipt,ifrmTrk);
  conf2u = trk2.pTrkconf_unet(ipt,ifrmTrk);

  % compute confs/prep
  tfisu1 = all(ptrk1==ptrk1u,1);
  tfisu2 = all(ptrk2==ptrk2u,1);
  fprintf(1,'isunet 1/2/both: %d/%d/%d\n',nnz(tfisu1),nnz(tfisu2),...
    nnz(tfisu1 & tfisu2));
    
  % compute RPE
  fprintf(1,'### RPE pt %d ###\n',ipt);
  tic;
  xytrk = cat(3,ptrk1,ptrk2);
  [X1,xyrp,rpe] = crobj.triangulate(xytrk);
  toc
  %prctile(rpe,[50 75 90 95 99])
  rpemn = mean(rpe,2);

  szassert(X1,[3 nfrm]);
  szassert(xyrp,[2 nfrm nview]);
  szassert(rpemn,[nfrm 1]);
  
  % conf tbl
  trpe = table(rpemn,conf1(:),conf1u(:),conf2(:),conf2u(:),...
    tfisu1(:),tfisu2(:),...
    ptrk1',ptrk2',X1',xyrp(:,:,1)',xyrp(:,:,2)',rpe(:,1),rpe(:,2),...
    'VariableNames',...
      {'rpemn' 'conf1' 'conf1u' 'conf2' 'conf2u' ...
       'isu1' 'isu2' ...
       'ptrk1' 'ptrk2' 'X1tri' 'ptrk1rp' 'ptrk2rp' 'ptrk1rpe' 'ptrk2rpe'});
  trpeconf{iipt} = trpe;
  assert(height(trpe)==nfrm);
  
  wbObj.startPeriod('Frames','shownumden',true,'denominator',nfrm);
  for ifrm=1:nfrm
    wbObj.updateFracWithNumDen(ifrm);
    
    trperow = trpe(ifrm,:);
   
    isspecial(ifrm,iipt) = trperow.rpemn>rpethresh;
    if isspecial(ifrm,iipt)

      % view pref; weight one view more based on conf
      lambda2 = 1;
      if doprefview
        iprefvw = preferviewCriteria(trperow,preferviewunetconf); 
        prefview(ifrm,iipt) = iprefvw;

        % TODO lambda2 vals will not allow comparison of scores between
        % prefervw frames and nonprefervw
        % lambdas only used when hmaps avail
        switch iprefvw
          case 0
            lambda2 = 1;
          case 1
            lambda2 = 1/preferviewlambda;
          case 2
            lambda2 = preferviewlambda;
          otherwise
            assert(false);
        end
      end
      
      if tfHM
        hm1 = gethmap(hmnr,hmnc,trperow.isu1,squeeze(hmbigrawfloat(ifrm,:,:,1,iipt)),...
          ptrk1(:,ifrm),mdnhmsigma2);
        hm2 = gethmap(hmnr,hmnc,trperow.isu2,squeeze(hmbigrawfloat(ifrm,:,:,2,iipt)),...
          ptrk2(:,ifrm),mdnhmsigma2);
        
        % hmap massage. issue is, unet and mdn hmaps are not very mutually
        % compatible when the rp error is high. in this case the
        % best/ultimate soln will tend to lie far away from the peak of the
        % distros. the unet hmap, being NN-generated tends to approach a low
        % but non-zero plateau away from hotspots/peaks. by contrast, the mdn
        % hmap being artificially generated vanishes exponentially like a
        % gaussian. in a naive comparison looking for solns far-from peaks,
        % the mdn hmap will dominate due to its unlimited paraboloid in log
        % space.
        
        % TODO this needs help with reconned hmaps
        
        % Note: this can cause loghm* to not be a real log(pdf) anymore as it
        % messes up the normalization.
        loghm1 = log(hm1);
        loghm2 = log(hm2);
        if trperow.isu1 && ~trperow.isu2 && ~unethmsarereconned
          unetmdn = median(loghm1(:));
          loghm2 = massageMdnHmap(loghm2,ptrk2(:,ifrm),unetmdn);
        elseif ~trperow.isu1 && trperow.isu2 && ~unethmsarereconned
          unetmdn = median(loghm2(:));
          loghm1 = massageMdnHmap(loghm1,ptrk1(:,ifrm),unetmdn);
        else
          % none; either unet/unet, or mdn/mdn, or all hmaps (unet and mdn
          % alike) are reconned. these hmaps are mutually comparable
        end
        
        roirad = max(trperow.rpemn,25);
        roi1 = [xyrp(1,ifrm,1)-roirad xyrp(1,ifrm,1)+roirad ...
                xyrp(2,ifrm,1)-roirad xyrp(2,ifrm,1)+roirad];
        roi2 = [xyrp(1,ifrm,2)-roirad xyrp(1,ifrm,2)+roirad ...
                xyrp(2,ifrm,2)-roirad xyrp(2,ifrm,2)+roirad];
        fprintf(1,'roirad is %.3f\n',roirad);
       
        [sbest,Xbest,xy1best,xy2best,roi1,roi2] = ...
          viewpref3dreconhmap(loghm1,loghm2,crobj,'dxyz',dxyz,'lambda2',lambda2,...
          'hm1roi',roi1,'hm2roi',roi2); 
        
        X3d(ifrm,:,iipt) = Xbest(:)';
        x2d(ifrm,:,1,iipt) = xy1best(:)';
        x2d(ifrm,:,2,iipt) = xy2best(:)';
        hmapscore(ifrm,iipt) = sbest;
        
        xall1 = [xy1best(1) ptrk1(1,ifrm) xyrp(1,ifrm,1)];
        yall1 = [xy1best(2) ptrk1(2,ifrm) xyrp(2,ifrm,1)];
        xall2 = [xy2best(1) ptrk2(1,ifrm) xyrp(1,ifrm,2)];
        yall2 = [xy2best(2) ptrk2(2,ifrm) xyrp(2,ifrm,2)];
        hmapscorecmp(ifrm,:,iipt) = ...
          interp2(loghm1,xall1,yall1,'nearest') + lambda2*interp2(loghm2,xall2,yall2,'nearest');
        x2dcmp(ifrm,:,1,1,iipt) = xall1(:);
        x2dcmp(ifrm,:,2,1,iipt) = yall1(:);
        x2dcmp(ifrm,:,1,2,iipt) = xall2(:);
        x2dcmp(ifrm,:,2,2,iipt) = yall2(:);
        
        hmaproi(ifrm,:,1,iipt) = roi1;
        hmaproi(ifrm,:,2,iipt) = roi2;        
      elseif doprefview && iprefvw>0
        % do view preference without HMs
        
        switch iprefvw
          case 1
            iotherview = 2;
          case 2
            iotherview = 1;
          otherwise
            assert(false);
        end
        
        xyprefer = xytrk(:,ifrm,iprefvw);
        xyother = xytrk(:,ifrm,iotherview);
          
        roiother = roisEPline(iotherview,:);
        [xEPL,yEPL] = crobj.computeEpiPolarLine(iprefvw,xyprefer,iotherview,roiother);

        % warn if the EP line is too widely spaced
        xyEPL = [xEPL(:) yEPL(:)];
        dxyEPL = diff(xyEPL,1,1);
        dzEPL = max(abs(dxyEPL),[],2);
        maxdzEPL = max(dzEPL);
        MAXDZEPL_WARN_THRESH = 1.0;
        if maxdzEPL > MAXDZEPL_WARN_THRESH
          warningNoTrace('EP line in view %d may be widely spaced: max dz=%.3f px.',...
            iotherview,maxdzEPL);
        end
        
        % select the point closest to xyother
        szassert(xyother,[2 1]);
        dEPLtrk = xyEPL-xyother';
        d2EPLtrk = sum(dEPLtrk.^2,2);
        [~,idx] = min(d2EPLtrk);
        xyEPLbest = xyEPL(idx,:);
        
        switch iprefvw
          case 1
            xy1best = xyprefer;
            xy2best = xyEPLbest';
          case 2            
            xy1best = xyEPLbest';
            xy2best = xyprefer;
          otherwise
            assert(false);
        end
        Xbest = crobj.triangulate(cat(3,xy1best(:),xy2best(:)));

        X3d(ifrm,:,iipt) = Xbest';
        x2d(ifrm,:,1,iipt) = xy1best';
        x2d(ifrm,:,2,iipt) = xy2best';

        xycmp = cat(3,...
                [xy1best'; xytrk(:,ifrm,1)'; xyrp(:,ifrm,1)'],...
                [xy2best'; xytrk(:,ifrm,2)'; xyrp(:,ifrm,2)']);
        x2dcmp(ifrm,:,:,:,iipt) = xycmp;
      else
        % no HM supplied, and no view preference. 
        
        X3d(ifrm,:,iipt) = X1(:,ifrm)';
        x2d(ifrm,:,:,iipt) = xyrp(:,ifrm,:);
      end
    else
      X3d(ifrm,:,iipt) = X1(:,ifrm)';
      x2d(ifrm,:,:,iipt) = xyrp(:,ifrm,:);
    end
  end
  wbObj.endPeriod();
  
end

wbObj.endPeriod();

function hmuse = gethmap(hmnr,hmnc,isunet,hmunetrawfloat,ptrk,mdnsigma2)
% hmbigrawfloat: float, neg and pos vals, no overall norm

if isunet
  hmuse = hmunetrawfloat+1; 
else
  xgv = 1:hmnc;
  ygv = 1:hmnr;
  [xg,yg] = meshgrid(xgv,ygv);
  assert(numel(ptrk)==2);
  assert(isscalar(mdnsigma2));
  mdnsigma2 = mdnsigma2*eye(2); % (co)variances
  hmuse = mvnpdf([xg(:) yg(:)],ptrk(:)',mdnsigma2);
  hmuse = reshape(hmuse,[hmnr hmnc]);
end
assert(all(hmuse(:)>=0));
hmuse = hmuse/sum(hmuse(:));

function loghm = massageMdnHmap(loghm,ptrk,mdnLogUnetHmap)
% loghm (in): orig log hmap
% ptrk: central pos
% mdnLogUnetHmap: median log(hmap) value for other/unet

LOGHM_FALLOF_PER_PX = 1e-4;

assert(numel(ptrk)==2);

tf = loghm<mdnLogUnetHmap;

[hmnr,hmnc] = size(loghm);
xgv = 1:hmnc;
ygv = 1:hmnr;
[xg,yg] = meshgrid(xgv,ygv);

r = sqrt( (xg-ptrk(1)).^2 + (yg-ptrk(2)).^2 );

loghm(tf) = mdnLogUnetHmap - LOGHM_FALLOF_PER_PX*r(tf);

function ivwpref = preferviewCriteria(trperow,unetconfthresh)
% ivwpref: 0 for no pref, or 1/2 for preferred view (1b)
%
% For a preference to occur, one view must be "high conf" and the other
% must be "low conf"

tfvw1hiconf = trperow.isu1 && trperow.conf1u>=unetconfthresh;
tfvw2hiconf = trperow.isu2 && trperow.conf2u>=unetconfthresh;

% TODO: maybe want a threshold on mdn conf too?
tfvw1loconf = ~trperow.isu1;
tfvw2loconf = ~trperow.isu2;

if tfvw1hiconf && tfvw2loconf
  ivwpref = 1;
elseif tfvw2hiconf && tfvw1loconf
  ivwpref = 2;
else
  ivwpref = 0;
end
