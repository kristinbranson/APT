function [suspscore,tblsusp,diagstr] = outlierCompute_Labels(lObj)

resp = inputdlg('Specify std threshold','',1,{'5'});
if isempty(resp)
  suspscore = [];
  tblsusp = [];
  diagstr = [];
  return;
end
stdthres = str2double(resp{1});

tblMFT = lObj.labelGetMFTableLabeled; % should be GT-aware

% pts [2,n,T] 2 = x,y, n = # of landmarks on each fly (17), T = # labeled
p = tblMFT.p;
mov = tblMFT.mov;
frm = tblMFT.frm;

[T,n] = size(p);
pts1 = reshape(p,[T,n/2,2]);
pts = permute(pts1,[3,2,1]);
mux = tblMFT.pTrx(:,1)';
muy = tblMFT.pTrx(:,2)';
theta = tblMFT.thetaTrx';

% align
pts_aligned = alignLandmarks(pts,mux,muy,theta);

% calculate each labels distance from mean for each landmark
[~,~,C] = size(pts_aligned);
meanx_pts = mean(pts_aligned(1,:,:),3);
meany_pts = mean(pts_aligned(2,:,:),3);
for i = 1:C
    err_pts(i,:) = sqrt(sum([(pts_aligned(1,:,i)-meanx_pts).^2;(pts_aligned(2,:,i)-meany_pts).^2],1));
end
stderr = std(err_pts);

%intialize tblsusp and suspscore
tblsusp = struct('mov',cell(0,1),'frm',[],'iTgt',[],'susp',[],'suspPt',[]);
suspscore = cellfun(@(x)ones(size(x,3),size(x,4)),lObj.labeledposGTaware,'uni',0);

for i = 1:numel(frm)
    [mstd, ldidx] = max(err_pts(i,:));
    if mstd > stderr(ldidx)*stdthres
        tblsusp(end+1,1).mov = double(tblMFT.mov(i).get());
        tblsusp(end).frm = tblMFT.frm(i);
        tblsusp(end).iTgt = tblMFT.iTgt(i);
        tblsusp(end).susp = round(mstd,2);
        tblsusp(end).suspPt = ldidx;
        
        iMov = tblMFT.mov(i).get();
        suspscore{iMov}(tblMFT.frm(i),tblMFT.iTgt(i)) = mstd;
    end
end

% convert to table
tblsusp = struct2table(tblsusp);
sortvars = {'mov'};
sortmode = {'ascend'};
tblsusp = sortrows(tblsusp,sortvars,sortmode);
diagstr = sprintf('std threshold = %.1f',stdthres);


