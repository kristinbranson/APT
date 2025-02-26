function moviesReorder(lObj,p)
% Reorder movies in an APT project
% 
% lObj: Labeler object
% p: permutation of 1:lObj.nmovies

warningNoTrace('This function is deprecated, please use the Labeler/movieReorder method.');

p = p(:);
if ~isequal(sort(p),(1:lObj.nmovies)')
  error('Input argument ''p'' must be a permutation of 1..%d.',lObj.nmovies);
end

% special cases
if ~isempty(lObj.tracker)
  error('Reordering is currently unsupported for projects with tracking.');
end
if ~isempty(lObj.suspScore) || ~isempty(lObj.suspSelectedMFT)
  error('Reordering is currently unsupported for projects with suspiciousness.');
end

iMov0 = lObj.currMovie;

% OK proceed
vcpw = lObj.viewCalProjWide;
if isempty(vcpw) || vcpw
  % none
else
  lObj.viewCalibrationData = lObj.viewCalibrationData(p);
end

FLDS = {'movieInfoAll' 'movieFilesAll' 'movieFilesAllHaveLbls'...
  'trxFilesAll'...
  'labeledpos' 'labeledposTS' 'labeledpostag' ... % 'labeledposMarked' 
  'labeledpos2'}; 
for f=FLDS,f=f{1}; %#ok<FXSET>
  lObj.(f) = lObj.(f)(p,:);
end

if ~lObj.gtIsGTMode  
  iMovNew = find(p==iMov0);
  lObj.movieSetGUI(iMovNew);
end
