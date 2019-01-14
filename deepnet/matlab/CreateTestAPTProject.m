function CreateTestAPTProject(fmoviefilelist,smoviefilelist,kinelistfile,savefile,varargin)
% CreateTestAPTProject(fmoviefilelist,smoviefilelist,kinelistfile,savefile,...)
% Creates an APT project with one video per fly from the input list of
% videos.
% Inputs:
% fmoviefilelist: Text file containing a list of front-view videos, one
% video per line
% smoviefilelist: Text file containing a list of side-view videos, one
% video per line. These videos should correspond line-by-line to those in
% fmoviefilelist. 
% kinelistfile: File containing look up information from fly number to
% calibration mat file
% savefile: Name of .lbl file to save project to. 
% Optional inputs:
% 'aptbasefile': path to lbl file containing a valid APT project. I set
% things up so that it loads in an apt file and then changes the video
% names so that I didn't have to get the project configuration right.
% Default value: 'fly178.lbl'
% 'nvideosperfly': number of videos to add to the project for each fly.
% Default value: 1
% 'projname': Name of the APT project. If not set, the name loaded from
% 'aptbasefile' will be used. 
% Example usage:
% CreateTestAPTProject('../View1Vids.txt','../View2Vids.txt','/groups/huston/hustonlab/flp-chrimson_experiments/fly2DLT_lookupTableStephen.csv','test.lbl','nvideosperfly',1,'projname','test')

[aptbasefile,nvideosperfly,projname] = myparse(varargin,...
	'aptbasefile','fly178.lbl',...
	'nvideosperfly',1,...
  'projname',[]);

% read in movie file names
fidf = fopen(fmoviefilelist,'r');
fmoviefiles = textscan(fidf,'%s');
fmoviefiles = fmoviefiles{1};
fclose(fidf);

fids = fopen(smoviefilelist,'r');
smoviefiles = textscan(fids,'%s');
smoviefiles = smoviefiles{1};
fclose(fids);

assert(numel(fmoviefiles) == numel(smoviefiles));

assert(all(cellfun(@exist,fmoviefiles)>0));
assert(all(cellfun(@exist,smoviefiles)>0));
nmovies = numel(fmoviefiles);

% read in kine data locs
ff = fopen(kinelistfile,'r');
kinematfiles = textscan(ff,'%d,%s');
fclose(ff);



% load in base apt data
ad = load(aptbasefile,'-mat');

% find data
trkfilefs = cell(1,nmovies);
trkfiless = cell(1,nmovies);
flynums = nan(1,nmovies);
istracked = false(1,nmovies);

for ndx = 1:nmovies
	
	% names of tracking files
  trkfilef = [fmoviefiles{ndx}(1:end-4),'.trk'];
  trkfiles = [smoviefiles{ndx}(1:end-4),'.trk'];
	
	if ~exist(trkfilef,'file') || ~exist(trkfiles,'file'),
    istracked(ndx) = false;
		warning('Trk files %s and/or %s do not exist',trkfilef,trkfiles);
    continue;
  end
  istracked(ndx) = true;
	
	trkfilefs{ndx} = trkfilef;
	trkfiless{ndx} = trkfiles;
	
	% fly number
  fparts = strsplit(fmoviefiles{ndx},filesep);  
  flynum = str2double(fparts{end-2}(4:6));
	flynums(ndx) = flynum;

	
end

fprintf('%d / %d files have tracking data\n',nnz(istracked),numel(istracked));

trkfilefs(~istracked) = [];
trkfiless(~istracked) = [];
fmoviefiles(~istracked) = [];
smoviefiles(~istracked) = [];
flynums(~istracked) = [];

% which flies
uniqueflynums = unique(flynums);

[ism,flyi2kineidx] = ismember(uniqueflynums,kinematfiles{1});
iskine = true(1,numel(trkfilefs));
if ~all(ism),
	warning('Did not find the following flies in the kinefile: %s',mat2str(uniqueflynums(~ism)));
  for i = find(~ism(:)'),
    iskine(uniqueflynums(i)==flynums) = false;
  end
  trkfilefs(~iskine) = [];
  trkfiless(~iskine) = [];
  fmoviefiles(~iskine) = [];
  smoviefiles(~iskine) = [];
  flynums(~iskine) = [];
  uniqueflynums = unique(flynums);
  [ism,flyi2kineidx] = ismember(uniqueflynums,kinematfiles{1});
  assert(all(ism));
end
fprintf('%d / %d files have kine data\n',nnz(iskine),numel(iskine));


nflies = numel(uniqueflynums);

adnew = ad;
adnew.movieFilesAll = cell(0,2);
adnew.movieInfoAll = cell(0,2);
adnew.trxFilesAll = cell(0,2);
adnew.viewCalibrationData = cell(0,1);
adnew.labeledpos = cell(0,1);
adnew.labeledpos2 = cell(0,1);
adnew.labeledpostag = cell(0,1);
adnew.labeledposTS = cell(0,1);
adnew.currMovie = 1;
adnew.currTarget = 1;
adnew.viewCalibrationData = {};
adnew.viewCalProjWide = false;
if ~isempty(projname),
  adnew.projname = projname;
end

expi = 1;
mr = MovieReader();

for flyi = 1:nflies,
	flynum = uniqueflynums(flyi);
	idxcurr = find(flynums==flynum);
  fprintf('Fly %d, found %d movies tracked\n',flynum,numel(idxcurr));
	ncurr = numel(idxcurr);
	if nvideosperfly == 1,
		fileidx = idxcurr(ceil(ncurr/2));
	elseif ncurr <= nvideosperfly,
		fileidx = idxcurr;
	else
		fileidx = idxcurr(round(linspace(1,ncurr,nvideosperfly)));
	end
	kinei = flyi2kineidx(flyi);
	kinematfile = kinematfiles{2}{kinei};

  caldata = CalRigSH();
  caldata.setKineData(kinematfile);
  
	for i = 1:numel(fileidx),
		filei = fileidx(i);
		fmoviefile = fmoviefiles{filei};
		smoviefile = smoviefiles{filei};
		adnew.movieFilesAll{expi,1} = fmoviefile;
		adnew.movieFilesAll{expi,2} = smoviefile;
    
    mr.open(fmoviefile);
    ifo = struct();
    ifo.nframes = mr.nframes;
    ifo.info = mr.info;
    mr.close();
    adnew.movieInfoAll{expi,1} = ifo;

    mr.open(smoviefile);
    ifo = struct();
    ifo.nframes = mr.nframes;
    ifo.info = mr.info;
    mr.close();
    adnew.movieInfoAll{expi,2} = ifo;

    adnew.trxFilesAll{expi,1} = '';
    adnew.trxFilesAll{expi,2} = '';
    adnew.viewCalibrationData{expi,1} = caldata;
    ftrx = load(trkfilefs{filei},'-mat');
    strx = load(trkfiless{filei},'-mat');
    npts = size(ftrx.pTrk,1);
    T = size(ftrx.pTrk,3);
    adnew.labeledpos2{expi,1} = nan([npts*2,2,T]);
    adnew.labeledpostag{expi,1} = cat(1,ftrx.pTrkTag,ftrx.pTrkTag);
    adnew.labeledposTS{expi,1} = cat(1,ftrx.pTrkTS,strx.pTrkTS);
    adnew.labeledposMarked{expi,1} = false([npts*2,T]);
    adnew.labeledpos{expi,1} = cat(1,ftrx.pTrk,strx.pTrk);
    
		expi = expi + 1;
	end
	
	
end

save(savefile,'-struct','adnew','-mat');