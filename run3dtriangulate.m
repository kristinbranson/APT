function run3dtriangulate(trkfile1,trkfile2,pptype,varargin)
% Run 3d postproc and resave to trkfiles in place
% 
% trkfile1/2: full paths trkfiles (these will be MODIFIED in-place)
% pptype: see PostProcess.triangulate

[crigmat,rois] = myparse(varargin,...
  'crigmat',[],... % path to mat-file containing single CalRig object to use 
  'rois',[1 720 1 540;1 720 1 540]... % for RF
  );

crig = loadSingleVariableMatfile(crigmat);
assert(isa(crig,'CalRig'));

fprintf(1,'Loaded CalRig from %s.\n',crigmat);

szassert(rois,[2 4]);

trk1 = load(trkfile1,'-mat');
trk2 = load(trkfile2,'-mat');

fprintf(1,'Loaded trkfiles \n%s\n%s\n',trkfile1,trkfile2);

%ld = loadLbl('/path/to/your/project.lbl');
%crig = ld.viewCalibrationData{1}; % to use calibration (or 'CalRig') 
% object for first movie (movie index 1) in that project

[trk1new,trk2new] = PostProcess.triangulate(trk1,trk2,rois,crig,pptype);
% trk1new, trk2new contain a subset of fields of the original trkfiles. You can "save -append" 
% the new structures to the old/existing trkfiles, but be careful, this will overwrite the .pTrk 
% field! The original data should be in the .pTrkSingleView field.

fprintf(1,'Triangulation complete...\n');

save(trkfile1,'-append','-struct','trk1new');
fprintf(1,'Saved/appended variables ''pTrkSingleView'', ''pTrk'', ''pTrk3d'' to trkfile %s.\n',...
  trkfile1);
save(trkfile2,'-append','-struct','trk2new');
fprintf(1,'Saved/appended variables ''pTrkSingleView'', ''pTrk'', to trkfile %s.\n',...
  trkfile2);