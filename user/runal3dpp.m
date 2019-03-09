function [X3d,x2d,trpeconf,isal,alpreferview,alscore,alscorecmp,alscorecmpxy,alroi] = ...
            runal3dpp_new(trk1,trk2,hmbig,crobj,varargin)

% TODO: preferviewnborrad

[ipts,frms,dxyz,mdnhmsigma2,unethmsarereconned,rpethresh,...
  preferview,preferviewunetconf,preferviewlambda,preferviewnborrad,wbObj] = myparse(varargin,...
  'ipts',1:2,...
  'frms',1:3000,... % CURRENTLY aASSUMED CONTINUOUS SEQ
  'dxyz',0.01,...
  'mdnhmsigma2',3.2^2,... % var for mdn fake hmap
  'unethmsarereconned',true,... % if false, we massage/flatten mdn hmaps to make them comparable to 'real' unet hmaps, which do not shrink to -inf in logspace
  'rpethresh',6,... % if rpemn><this>, then do the fancy 3dgridRP. specify as inf to never do it
  'preferview',true,... % if true, when doing the fancy, if one view has more conf then prefer it
  'preferviewunetconf',0.7,...
  'preferviewlambda',3,... % used when preferview==true
  'preferviewnborrad',2,... % used when preferview==true. in 'weak' view, include nboring frames with this rad. use 0 for no-nbors
  'wbObj',WaitBarWithCancelCmdline('3dpp')  ...
);

npts = numel(ipts);
nfrm = numel(frms);
nview = 2;

tfhmidx = ismember(double(hmbig.frmagg),frms);
assert(isequal(double(hmbig.frmagg(tfhmidx)),frms(:)));
hmbigrawfloat = hmbig.hmagg(tfhmidx,:,:,:,:);
[~,hmnr,hmnc,~,~] = size(hmbigrawfloat);
szassert(hmbigrawfloat,[nfrm hmnr hmnc nview npts]); % XXX assumes hmbigarr already selected for ipts

X3d = nan(nfrm,3,npts);
x2d = nan(nfrm,2,nview,npts);
trpeconf = cell(npts,1); % trpeconf{ipt} is [nfrm] table
isal = false(nfrm,npts);
alpreferview = zeros(nfrm,npts); % 0 = no pref, etc
alscore = nan(nfrm,npts);
alscorecmp = nan(nfrm,3,npts); % best3d, orig, origRPs
alscorecmpxy = nan(nfrm,3,2,nview,npts); % x-y for alscorecmp
alroi = nan(nfrm,4,nview,npts);
origTrkStuff = cell(npts,1);

wbObj.startPeriod('Points','shownumden',true,'denominator',npts);

for iipt=1:npts
  wbObj.updateFracWithNumDen(iipt);

  ipt = ipts(iipt);  

  assert(isequal(trk1.pTrkFrm(frms),frms));
  assert(isequal(trk2.pTrkFrm(frms),frms));
  
  ptrk1 = squeeze(trk1.pTrk(ipt,:,frms)); % 2xnfrm
  ptrk2 = squeeze(trk2.pTrk(ipt,:,frms)); % 2xnfrm
  ptrk1u = squeeze(trk1.pTrklocs_unet(ipt,:,frms))+1; % XXX PLUS 1
  ptrk2u = squeeze(trk2.pTrklocs_unet(ipt,:,frms))+1; % XXX PLUS 1
  conf1 = trk1.pTrkconf(ipt,frms); % 1xnfrm
  conf1u = trk1.pTrkconf_unet(ipt,frms); % etc
  conf2 = trk2.pTrkconf(ipt,frms);
  conf2u = trk2.pTrkconf_unet(ipt,frms);

  % compute confs/prep
  tfisu1 = all(ptrk1==ptrk1u,1);
  tfisu2 = all(ptrk2==ptrk2u,1);
  tfisunet = tfisu1 & tfisu2;
  fprintf(1,'isunet 1/2/both: %d/%d/%d\n',nnz(tfisu1),nnz(tfisu2),nnz(tfisunet));
    
  % compute RPE
  fprintf(1,'### RPE pt %d ###\n',ipt);
  tic;
  [X1,xp1rp,xp2rp,rpe1,rpe2] = crobj.stereoTriangulate(ptrk1,ptrk2);
  toc
  rpe = [rpe1(:) rpe2(:)];
  prctile(rpe,[50 75 90 95 99])
  rpemn = mean(rpe,2);

  szassert(X1,[3 nfrm]);
  szassert(xp1rp,[2 nfrm]);
  szassert(xp2rp,[2 nfrm]);
  szassert(rpemn,[nfrm 1]);
  
  % conf tbl
  trpe = table(rpemn,conf1(:),conf1u(:),conf2(:),conf2u(:),...
    tfisu1(:),tfisu2(:),...
    ptrk1',ptrk2',X1',xp1rp',xp2rp',rpe1(:),rpe2(:),...
    'VariableNames',...
    {'rpemn' 'conf1' 'conf1u' 'conf2' 'conf2u' 'isu1' 'isu2' 'ptrk1' 'ptrk2' 'X1strotri' 'xp1rp' 'xp2rp' 'rpe1' 'rpe2'});
  trpeconf{iipt} = trpe;
  assert(height(trpe)==nfrm);
  
  wbObj.startPeriod('Frames','shownumden',true,'denominator',nfrm);
  for ifrm=1:nfrm
    wbObj.updateFracWithNumDen(ifrm);
    
    trperow = trpe(ifrm,:);
   
    isal(ifrm,iipt) = trperow.rpemn>rpethresh;
    if isal(ifrm,iipt)

      % view pref; weight one view more based on conf
      lambda2 = 1;
      if preferview
        iprefervw = preferviewCriteria(trperow,preferviewunetconf); 
        alpreferview(ifrm,iipt) = iprefervw;

        % TODO lambda2 vals will not allow comparison of scores between
        % prefervw frames and nonprefervw
        switch iprefervw
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
      
      % what are our hmaps?
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
      roi1 = [xp1rp(1,ifrm)-roirad xp1rp(1,ifrm)+roirad ...
              xp1rp(2,ifrm)-roirad xp1rp(2,ifrm)+roirad];
      roi2 = [xp2rp(1,ifrm)-roirad xp2rp(1,ifrm)+roirad ...
              xp2rp(2,ifrm)-roirad xp2rp(2,ifrm)+roirad];
      fprintf(1,'roirad is %.3f\n',roirad);

      if ifrm==198
        disp('asd');
      end

     [sbest,Xbest,xy1best,xy2best,roi1,roi2] = ...
        al3dpp(loghm1,loghm2,crobj,'dxyz',dxyz,'lambda2',lambda2,'hm1roi',roi1,'hm2roi',roi2);
      
      
      X3d(ifrm,:,iipt) = Xbest(:)';
      x2d(ifrm,:,1,iipt) = xy1best(:)';
      x2d(ifrm,:,2,iipt) = xy2best(:)';
      alscore(ifrm,iipt) = sbest;
      
      xall1 = [xy1best(1) ptrk1(1,ifrm) xp1rp(1,ifrm)];
      yall1 = [xy1best(2) ptrk1(2,ifrm) xp1rp(2,ifrm)];
      xall2 = [xy2best(1) ptrk2(1,ifrm) xp2rp(1,ifrm)];
      yall2 = [xy2best(2) ptrk2(2,ifrm) xp2rp(2,ifrm)];
      alscorecmp(ifrm,:,iipt) = ...
        interp2(loghm1,xall1,yall1,'nearest') + lambda2*interp2(loghm2,xall2,yall2,'nearest');
      alscorecmpxy(ifrm,:,1,1,iipt) = xall1(:);
      alscorecmpxy(ifrm,:,2,1,iipt) = yall1(:);
      alscorecmpxy(ifrm,:,1,2,iipt) = xall2(:);
      alscorecmpxy(ifrm,:,2,2,iipt) = yall2(:);
      
      alroi(ifrm,:,1,iipt) = roi1;
      alroi(ifrm,:,2,iipt) = roi2;
    else
      X3d(ifrm,:,iipt) = X1(:,ifrm)';
      x2d(ifrm,:,1,iipt) = xp1rp(:,ifrm)';
      x2d(ifrm,:,2,iipt) = xp2rp(:,ifrm)'; 
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
