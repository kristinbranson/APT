function CPRLabelTrackerTrack(tObjFile,resFileBase,movFiles,frmSpec)
% Track movies using saved/trained CPRLabelTracker
%
% tObjFile: full path to file containing CPRLabelTracker object. File must
%   contain a single variable (the object)
% resFileBase: full path, basename results file to be saved
% movs: cellstr, full paths for movies to track. OR, char file containing
%   movies to track
% frmSpec: currently, an integer specifying df (frame spacing) for tracked
% results

if ischar(movFiles) && exist(movFiles,'file')>0
  movFiles = importdata(movFiles);
end
if ischar(frmSpec)
  frmSpec = str2double(frmSpec);
end
  
%#function CPRLabelTracker
t = load(tObjFile);
flds = fieldnames(t);
assert(isscalar(flds));
t = t.(flds{1});

validateattributes(frmSpec,{'numeric'},{'positive' 'integer' 'scalar'});
df = frmSpec;

D = size(t.trnDataTblP.p,2);

nMov = numel(movFiles);
for iMov = 1:nMov
  movieFull = movFiles{iMov};
  [~,movieShrt] = myfileparts(movieFull);
  
  vr = VideoReader(movieFull); %#ok<TNMLP>
  nf = get(vr,'NumberOfFrames');
  frm = 1:df:nf;
  frm = frm(:);
  nrows = numel(frm);
  
  mov = repmat({movieFull},nrows,1);
  movS = repmat({movieShrt},nrows,1);
  p = nan(nrows,D); % dummy label positions; will not be used
  tfocc = false(nrows,D/2);
  
  tblP = table(mov,movS,frm,p,tfocc);
  t.initData();
  t.track([],[],'useRC',1,'tblP',tblP);  
  [~,movSSansExt] = fileparts(movieShrt);
  resFile = [resFileBase '_' movSSansExt '.mat'];
  t.saveTrackRes(resFile);  
  t.initTrackRes();
end

