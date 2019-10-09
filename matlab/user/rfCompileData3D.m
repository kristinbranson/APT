function [I,pGT3d,bboxes,pGt3dRCerr,tMFP] = rfCompileData3D(tMFP,lblMovSets,lblLpos,crig2)
% rfCompileData3D -- Compile 3D training data
%
% tMFP: movieset/frame/pts table, [N] rows
% * (each row of) movs is a [nView] cellstr containing full movienames for each view
% * (") frm is a scalar frame number
% * (") ipts2VwLblNO: [nPts2VwLblNO] array of pt indices that are labeled
% in 2 views, NO for this movieset/frame
% * (") iVwsLblNOCode: [nPts2VwLblNO] cellstr indicating which views are
% labeled for each point in ipts2VwLblNO, eg 'lr', 'lrb', etc
% 
% lblMovSets: From .lbl file. [nMovSet x nView] movie fullpaths
%
% lblLpos: From .lbl file. [nMovSetx1] cell array, each el of which is 
% [19*3x2xnfrm] labeledpos array for 2d labels. Order of 1st dim is
% [pt1_vw1 pt2_vw1 ... pt19_vw1 pt1_vw2 ... pt19_vw2 pt1_vw3 ... pt19_vw3]
%
% crig: CalibratedRig2 object. Currently applies to all moviesets
%
%
% I: [NxnView] images
% pGT3d: [Nx19*3] Reconstructed GT shapes, 3d coords, in 'L' camera coord 
%   sys. Order is [x1 x2 .. x19 y1 y2 .. y19 z1 z2 .. z19]. Reconstruction
%   is done for pts that have labels (NO) in either LB, RB, or LRB. Other
%   points, including points with LR labels, will have nan coords.
% bboxes:
% pGT3dRCerr: struct with fields .L, .R, .B; Each pGT3dRCerr.(fld) is
% [Nx19] giving recon error computed by projecting onto that view.
% tMFP (out): augmented with iMS (iMovSet)


% NOTE: for p-vectors or shapes, there are two flavors:
% * "concatenated-projected", ie you take the projected labels and
% concatenate. numel here is npts x nView x 2, raster order is pt, view,
% coord (x vs y).
% * absolute/3d, numel here is npts x 3, raster order is pt, coord (x vs y
% vs z).

tMFPflds = tMFP.Properties.VariableNames;
tf = strncmp('npts',tMFPflds,4);
assert(nnz(tf)==1);
fldNptsLbled = tMFPflds{tf};
tf = strncmp('ipts',tMFPflds,4);
assert(nnz(tf)==1);
fldIptsLbled = tMFPflds{tf};
tf = ~cellfun(@isempty,regexpi(tMFPflds,'code','once'));
assert(nnz(tf)==1);
fldCodeLbled = tMFPflds{tf};

str = sprintf('Found fields in tMFP: (npts,ipts,code)->(%s,%s,%s). OK?.',...
  fldNptsLbled,fldIptsLbled,fldCodeLbled);
btn = questdlg(str,'Confirm table fields','OK','Cancel','OK');
if isempty(btn)
  btn = 'Cancel';
end
switch btn
  case 'OK'
    % none
  case 'Cancel'
    return;
end
    


%%
nRows = size(tMFP,1);
fprintf(1,'Number of rows: %d\n',nRows);

%% nView
nView = size(tMFP.movs,2);
fprintf(1,'Number of views: %d\n',nView);

%% moviesets
nMovSet = size(lblMovSets,1);
szassert(lblMovSets,[nMovSet nView]);
szassert(lblLpos,[nMovSet 1]);
nFrames = nan(nMovSet,1);
for iMovSet=1:nMovSet
  lpos = lblLpos{iMovSet};
  nFrames(iMovSet) = size(lpos,3);
  szassert(lpos,[57 2 nFrames(iMovSet)]);
  fprintf(1,'Movieset %d: %d frames (from labeledpos)\n',iMovSet,nFrames(iMovSet));
end

%% Label codes, distributions
codes = cat(1,tMFP.(fldCodeLbled){:});
codes = categorical(codes);
fprintf(1,'Codes summary: \n');
summary(codes);

%% recon 3d positions; augment table 
pGT3d = nan(nRows,19*3);
pGt3dRCerr.L = nan(nRows,19);
pGt3dRCerr.R = nan(nRows,19);
pGt3dRCerr.B = nan(nRows,19);
iMS = nan(nRows,1);
for iRow = 1:nRows
  
  movs = tMFP.movs(iRow,:);
  tfFound = false;
  for iMovSet=1:nMovSet
    if isequal(movs,lblMovSets(iMovSet,:))
      tfFound = true;
      break;
    end
  end
  assert(tfFound,'Row %d: .movs not found in lblMovSets.\n',iRow);
  iMS(iRow) = iMovSet;
  
  lpos = reshape(lblLpos{iMovSet},[19 nView 2 nFrames(iMovSet)]);

  frm = tMFP.frm(iRow);
  nptsLbled = tMFP.(fldNptsLbled)(iRow);
  iptsLbled = tMFP.(fldIptsLbled){iRow};
  iVwCodeLbled = tMFP.(fldCodeLbled){iRow};
  szassert(iptsLbled,[nptsLbled 1]);
  szassert(iVwCodeLbled,[nptsLbled 1]);  
  
  XLrow = nan(3,19);
  for j = 1:nptsLbled
    ipt = iptsLbled(j);
    lposPt = squeeze(lpos(ipt,:,:,frm));
    szassert(lposPt,[3 2]);
    code = iVwCodeLbled{j};
    yL = lposPt(1,[2 1]);
    yR = lposPt(2,[2 1]);
    yB = lposPt(3,[2 1]);
    switch code
      case 'lr'
        %assert(all(isnan(yB(:))));
        XLrow(:,ipt) = crig2.stereoTriangulateCropped(yL,yR,'L','R');        
      case 'lb'
        %assert(all(isnan(yR(:))));
        XLrow(:,ipt) = crig2.stereoTriangulateLB(yL,yB);        
      case 'rb'
        %assert(all(isnan(yL(:))));
        XBbr = crig2.stereoTriangulateBR(yB,yR);
        XLrow(:,ipt) = crig2.camxform(XBbr,'bl');
      case 'lrb'
        [~,~,~,...
          pGt3dRCerr.L(iRow,ipt),pGt3dRCerr.R(iRow,ipt),pGt3dRCerr.B(iRow,ipt),...
          ~,~,~,XLrow(:,ipt)] = crig2.calibRoundTripFull(yL,yR,yB);
    end
  end
  
  tmp = XLrow'; % [19x3]
  pGT3d(iRow,:) = tmp(:);
  
  if mod(iRow,10)==0
    fprintf(1,'Reconstructed 3D pt: row %d\n',iRow);
  end
end

tMFP = [tMFP table(iMS)];

%% I, bboxes

% open moviereaders
mrcell = cell(size(lblMovSets));
for iMovSet=1:nMovSet
  for iView=1:nView
    mr = MovieReader();
    mr.open(lblMovSets{iMovSet,iView});
    mr.forceGrayscale = true;
    mrcell{iMovSet,iView} = mr;
  end
end

I = cell(nRows,nView);
for iRow=1:nRows
  frm = tMFP.frm(iRow);
  iMovSet = tMFP.iMS(iRow);
  for iView=1:nView
    I{iRow,iView} = mrcell{iMovSet,iView}.readframe(frm);
  end
  if mod(iRow,10)==0
    fprintf(1,'Read images: row %d\n',iRow);
  end
end


%% bboxes
d = 3;
pGTtmp = reshape(pGT3d,nRows,19,d);
pGTcoordmins = nan(1,d);
pGTcoordmaxs = nan(1,d);
for iCoord=1:d
  z = pGTtmp(:,:,iCoord); % x-, y-, or z-coords for all rows, pts
  pGTcoordmins(iCoord) = nanmin(z(:));
  pGTcoordmaxs(iCoord) = nanmax(z(:));
end
dels = pGTcoordmaxs-pGTcoordmins;
% pad by 50% in every dir
pads = dels/2;
widths = 2*dels; % del (shapes footprint) + 2*pads (one on each side)
bboxes = [pGTcoordmins-pads widths];
