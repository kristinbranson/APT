function cleanJanLbl(lblfile)
% Clean label file for Jan data
% 
% lblfile: lblfile to be cleaned. MAKE A COPY, THIS WILL BE OVERWRITTEN!

l = load(lblfile,'-mat');
assert(numel(l.movieFilesAll)==1);
mov = l.movieFilesAll{1};
lpos = l.labeledpos{1};
lpostag = l.labeledpostag{1};
[npt,~,nfrm] = size(lpos);

% movie
if exist(mov,'file')==0
  warnstr = sprintf('Cannot find movie ''%s''. Please browse to movie location.',mov);
  warndlg(warnstr,'Missing movie','modal');
  pause(1);
  lastmov = RC.getprop('lbl_lastmovie');
  if isempty(lastmov)
    lastmov = pwd;
  end
  %pth = fileparts(lblfile);
  [newmovfile,newmovpath] = uigetfile('*.*','Select movie',lastmov);
  if isequal(newmovfile,0)
    error('Labeler:mov','Cannot find movie ''%s''.',mov);
  end  
  mov = fullfile(newmovpath,newmovfile);
  RC.saveprop('lbl_lastmovie',mov);
end
l.movieFilesAll{1} = mov;

% overview
nptsLbled = Labeler.labelPosNPtsLbled(lpos);
tfLbled = nptsLbled>0;
fLbled = find(tfLbled);
nptsLblUn = unique(nptsLbled);
nptsLblUnCnt = arrayfun(@(x)nnz(nptsLbled==x),nptsLblUn);
fprintf(1,'npt=%d, nfrm=%d\n',npt,nfrm);
fprintf(1,'[nptsLblUn nptsLblUnCnt]:\n');
disp([nptsLblUn nptsLblUnCnt]);
pause(10);

% flip 
mr = MovieReader;
mr.open(mov);
fLbled1 = fLbled(1);
im = mr.readframe(fLbled1);
hFig = figure('windowstyle','docked');
imagesc(im);
axis image
colormap gray
hold on
title(sprintf('First lbled frame: %d',fLbled1),'fontweight','bold','interpreter','none');
lpos1 = lpos(:,:,fLbled1);
clrs = jet(npt);
for i = 1:npt
  plot(lpos1(i,1),lpos1(i,2),'o','MarkerSize',10,'MarkerFaceColor',clrs(i,:),'Color',clrs(i,:));
end
btn = questdlg('Flip labels u/d?','Flip?','Yes, flip','No','Cancel','No');
if isempty(btn)
  btn = 'Cancel';
end
switch btn
  case 'Yes, flip'
    assert(size(im,1)==256);
    assert(size(lpos,2)==2);
    lpos(:,2,:) = 256-lpos(:,2,:);
    l.labeledpos{1} = lpos;
    fprintf(1,'Labels flipped u/d\n');
  case 'No'
  case 'Cancel'
    error('Labeler:mov','User canceled. No changes saved.');
end
pause(2);
close(hFig);

% copy pts 123 from first pt to others
f0 = find(nptsLbled==npt);
assert(isscalar(f0),'Expected exactly one pt with all %d labeled pts.',npt);
fset = setdiff(fLbled,f0);
for f=fset(:)'
  tmp = lpos(1:3,:,f);
  if ~all(isnan(tmp(:)))
    warning('cleanJanLbl:existing123','Unexpected labeled pts123 exist, frm %d',f);
  end
  lpos(1:3,:,f) = lpos(1:3,:,f0);
end
fprintf(1,'Copied pts 1,2,3 from frame f0=%d to %d other frames.\n',f0,numel(fset));


% clear partially-labeled frames
nptsLbled = Labeler.labelPosNPtsLbled(lpos);
%tf0 = nPtsLbled==0;
%tfGood = nPtsLbled==npt;
tfRest = ismember(nptsLbled,1:npt-1);
nRest = nnz(tfRest);
lpos(:,:,tfRest) = nan;
lpostag(:,tfRest) = {[]};
fprintf(1,'cleared %d partially-labeled frames:\n',nRest);
disp(find(tfRest));
  
% overview
nptsLbled = Labeler.labelPosNPtsLbled(lpos);
%tfLbled = nptsLbled>0;
%fLbled = find(tfLbled);
nptsLblUn = unique(nptsLbled);
nptsLblUnCnt = arrayfun(@(x)nnz(nptsLbled==x),nptsLblUn);
fprintf(1,'npt=%d. [nptsLblUn nptsLblUnCnt]:\n',npt);
disp([nptsLblUn nptsLblUnCnt]);

l.labeledpos{1} = lpos;
l.labeledpostag{1} = lpostag;
fprintf(1,'(Re)saving to %s\n',lblfile);
save(lblfile,'-mat','-struct','l');

lObj = Labeler();
lObj.projLoadGUI(lblfile);
fprintf(1,'Opening proj %s in Labeler. Close Labeler to finish.\n',lblfile);
uiwait(lObj.gdata.mainFigure_);