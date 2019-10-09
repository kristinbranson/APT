% Session holds the state of the Camera Calibration App
%
%   This class holds the entire state of the camera calibration UI.
%   It is used to save and load the camera calibration session. It is also
%   used to pass data amongst other classes.

% Copyright 2012-2013 The MathWorks, Inc.

classdef Session < handle

    properties
        CameraModel;              % holds all of the calibration options
        OptimizationOptions = [];      % initial values for intrinsics and radial distortion
        CameraParameters = [];    % actual calibration results
        EstimationErrors = [];
        ShouldExportErrors = false;
        
        BoardSet = [];            % holds all checkerboard information
        
        HasEnoughBoards = false;  % true if enough images were loaded and processed
        CanExport = false;        % true when cameraParameters can be exported
        IsChanged = false;        % true when session may need saving
        
        ExtrinsicsView = 'CameraCentric';
        
        % ErrorsView is no longer used, since we only use the BarGraph view
        % in the app. However, we have to keep this property around for
        % compatibility with older versions.
        ErrorsView = 'BarGraph';
        
        Filename = []; % filename for the session
        
        ExportVariableName = 'cameraParams'; % default export variable name     
        ExportErrorsVariableName = 'estimationErrors';
    end
    
    properties(Access=private, Hidden)
        Version = ver('vision');
    end
        
    properties(Dependent)
        FileName; % Had to add this for backward compatibility
        
        % The following properties are NOT mutually exclusive.
        % An empty session (with empty BoardSet and CameraParameters) can 
        % be either stereo or single camera. 
        IsValidStereoCameraSession;
        IsValidSingleCameraSession;
    end
    
    methods
        
        %------------------------------------------------------------------
        function fileName = get.FileName(this)
            fileName = this.Filename;
        end
        
        %------------------------------------------------------------------
        function set.FileName(this, fileName)
            this.Filename = fileName;
        end
        
        %------------------------------------------------------------------
        function tf = get.IsValidStereoCameraSession(this)
            tf = (isempty(this.CameraParameters) || ...
                  isa(this.CameraParameters, 'stereoParameters')) && ...
                 (isempty(this.BoardSet) || ...
                  size(this.BoardSet.FullPathNames, 1) == 2);
        end
        
        %------------------------------------------------------------------
        function tf = get.IsValidSingleCameraSession(this)
            tf = (isempty(this.CameraParameters) || ...
                  isa(this.CameraParameters, 'cameraParameters')) && ...
                 (isempty(this.BoardSet) || ...
                  size(this.BoardSet.FullPathNames, 1) == 1);
        end
            
        %------------------------------------------------------------------
        % return true if the tool went though the calibration
        %------------------------------------------------------------------
        function ret = isCalibrated(this)
            ret = ~isempty(this.CameraParameters);
        end

        function ret = hasAnyBoards(this)
            ret = ~isempty(this.BoardSet) && ...
                    this.BoardSet.NumBoards ~= 0;
        end        
        
        %------------------------------------------------------------------
        function reset(this)
            
            this.ExtrinsicsView = 'CameraCentric';
            this.FileName = [];
            this.CanExport = false;
            this.HasEnoughBoards = false;
            this.CameraParameters = [];
            this.IsChanged = false;
            this.CameraModel = [];
            this.OptimizationOptions = [];
            
            if ~isempty(this.BoardSet)
                this.BoardSet.reset();
            end
        end
        
        %------------------------------------------------------------------
        % Wipes only the calibration portion of the session
        %------------------------------------------------------------------
        function resetCalibration(this)
            this.CanExport = false;
            this.CameraParameters = [];
        end
        
        %------------------------------------------------------------------
        function checkImagePaths(this, pathname, filename)
            if ~isempty(this.BoardSet)
                this.BoardSet.checkImagePaths(pathname, filename);
            end
        end
        
        %------------------------------------------------------------------

    end
    
    methods (Static)
      function res = hlpOrthoSingleCalib(camidx,camSess)
        camParams = camSess.CameraParameters;
        nCalIm = camParams.NumPatterns;
        worldPts = camParams.WorldPoints;
        bs = camSess.BoardSet;
        imPtsUV = bs.BoardPoints;
        
        p0 = OrthoCam.p0fromRsTs(...
          permute(camParams.RotationMatrices,[2 1 3]),...
          camParams.TranslationVectors);
        
        fprintf(1,'NEW ADDITION June 2018. Never needed iterations on single-cam calib before.\n');
        fprintf(1,'Calibrating camera %d...]n',camidx);
        pause(1);        
        pOpt = p0;
        while 1
          pOpt = OrthoCam.calibrate1cam(nCalIm,worldPts,imPtsUV,pOpt);          
          STOP = 'Stop optimization, looks good';
          RESTART = 'Restart optimization';
          CANCEL = 'Cancel';
          resp = questdlg('Restart optimization?','Optimization waypoint',...
            STOP,RESTART,CANCEL,RESTART);
          if isempty(resp)
            resp = CANCEL;
          end
          switch resp
            case STOP
              break;
            case RESTART
              % none; while loop will proceed
            case CANCEL
              error('Session:cal','Calibration canceled.');
          end
        end
        fprintf(1,'OK, done.\n');
        pause(1);        
        
        res = struct();
        res.p0 = p0;
        res.pOpt = pOpt;
        %res.oFcn = oFcn;
        res.nPat = nCalIm;
        res.worldPts = worldPts;
        res.imPtsUV = imPtsUV;
        res.boardSetFPNs = bs.FullPathNames;
        res.ts = now();        
      end
      function dRP = hlpAssessOrthoSingleCalib(camSess,pOpt)
        
        camParams = camSess.CameraParameters;
        nCalIm = camParams.NumPatterns;
        worldPoints = camParams.WorldPoints;
        nPts = size(worldPoints,1);
        szassert(worldPoints,[nPts 2]);
        worldPtsXYZ = [worldPoints zeros(nPts,1)]';
        
        bs = camSess.BoardSet;
        imPtsUV = bs.BoardPoints;
        szassert(imPtsUV,[nPts 2 nCalIm]);        
        calibImPts = permute(imPtsUV,[2 1 3]);
      
        oFcn = @(p)OrthoCam.oFcn(p,nCalIm,worldPtsXYZ,calibImPts);
        dRP = oFcn(pOpt);
        dRP = reshape(dRP,[nPts nCalIm]);
      end
      function [calRes,calResFile] = hlpLoadSingleOrthoCamCalRes(camidx)
        orthopath = getpref('orthocam','lastpath',pwd);
        str = sprintf('Select saved single-Orthocam calibration for Camera%d.',camidx);
        [fname,pth] = uigetfile('*.mat',str,orthopath);
        if isequal(fname,0)
          calRes = [];
          calResFile = '';
        else
          calResFile = fullfile(pth,fname);
          calRes = load(calResFile,'-mat');
          setpref('orthocam','lastpath',pth);
        end
      end
      function calResFile = hlpSaveSingleOrthoCamCalRes(calres,camsessmatfile,camidx) %#ok<INUSL>
        [sesspath,sessfile] = fileparts(camsessmatfile);
        calResFileProposed = fullfile(sesspath,[sessfile '_orthocam_monocal.mat']);
        str = sprintf('Save single-Orthocam calibration for Camera%d.',camidx);
        [fname,pth] = uiputfile(calResFileProposed,str);
        if isequal(fname,0)
          calResFile = '';
        else
          calResFile = fullfile(pth,fname);
          save(calResFile,'-struct','calres');
        end
      end
    end
    methods
      function [sess,sessMatfile] = hlpLoadAndCheckSingleCamSession(this,camidx)
        orthopath = getpref('orthocam','lastpath',pwd);
        str = sprintf('Select Camera%d Session/MAT-file saved from MATLAB Camera Calibrator App',camidx);
        [fname,pth] = uigetfile('*.mat',str,orthopath);
        if isequal(fname,0)
          sess = [];
          sessMatfile = '';
        else
          sessMatfile = fullfile(pth,fname);
          sess = load(sessMatfile,'-mat');
          setpref('orthocam','lastpath',pth);
          sess = sess.calibrationSession;
          fpn1 = this.BoardSet.FullPathNames(camidx,:)';
          fpn2 = sess.BoardSet.FullPathNames';
          if ispc
            fpn1 = lower(fpn1);
            fpn2 = lower(fpn2);
          end
          tf = ismember(fpn1,fpn2);
          if ~all(tf)
            error('Session:cal','One or more %s calibration images are not present in single-camera session: %s\n',...
              ['cam' num2str(camidx)],sessMatfile);
          end
        end
      end
    end
    methods (Static)
      function [r2vecs,t2vecs] = hlpSelSingleCamExtrinsics(iCam,res,bset)
        pOpt = res.pOpt;
        nCalIm = res.nPat;
        fpnsMono = res.boardSetFPNs(:);
        szassert(fpnsMono,[nCalIm 1]);
        fpnsStro = bset.FullPathNames(iCam,:)';
        if ispc
          fpnsStro = lower(fpnsStro);
          fpnsMono = lower(fpnsMono);
        end
        [tf,loc] = ismember(fpnsStro,fpnsMono);
        assert(all(tf));
        fprintf(1,'Selecting %d (rvec,tvec) extrinsic pairs out of %d from cam%d extrinsics:\n',...
          numel(fpnsStro),nCalIm,iCam);

        [~,~,~,~,~,~,r2vecs,t2vecs] = OrthoCam.unpack1cam(pOpt,nCalIm);
        r2vecs = r2vecs(:,loc);
        t2vecs = t2vecs(:,loc);        
      end
    end
    methods      
      function imagesUsed = calibrate(this)
        
        if this.IsValidSingleCameraSession && ~this.IsValidStereoCameraSession
          imagesUsed = this.calibrateOrig();
          return;
        end
        
        mlVer = ver('matlab');
        if ~strcmp(mlVer.Release,'(R2016b)')
          error('Session:cal',...
            'MATLAB version R2016b is required for Orthocam calibration.');
        end
        
        info = struct();                
        LOAD = 'Load existing orthocam calibrations';
        CALIB = 'Perform orthocam calibrations now';
        CANCEL = 'Cancel';
        resp = questdlg('Do you have existing single-camera OrthoCam calibrations for these two cameras? The calibrations must include all images included in this Stereo calibration project.',...
          'Load existing OrthoCam calibrations (single-cam)',...
          LOAD,CALIB,CANCEL,CALIB);
        if isempty(resp)
          resp = CANCEL;
        end
        info.monoCalOrigin = 'none'; % {'none','loaded','run'}
        switch resp
          case LOAD
            [info.cam1calres,info.cam1calresfile] = this.hlpLoadSingleOrthoCamCalRes(1);
            [info.cam2calres,info.cam2calresfile] = this.hlpLoadSingleOrthoCamCalRes(2);
            if isempty(info.cam1calres) || isempty(info.cam2calres)
              resp2 = questdlg('Do you want to perform OrthoCam calibrations now?',...
                'Perform OrthoCam calibrations','Yes','No','Yes');
              if isempty(resp2)
                resp2 = 'No';
              end
              switch resp2
                case 'Yes'
                  % none
                case 'No'
                  error('Session:cal','Calibration canceled.');
              end
              info.monoCalOrigin = 'run';
            else
              info.monoCalOrigin = 'loaded';
            end
          case CALIB
            info.monoCalOrigin = 'run';
          case CANCEL
            error('Session:cal','Calibration canceled.');
        end
        switch info.monoCalOrigin
          case 'run'
            % Run it
            [info.cam1Sess,info.cam1SessMatfile] = this.hlpLoadAndCheckSingleCamSession(1);
            [info.cam2Sess,info.cam2SessMatfile] = this.hlpLoadAndCheckSingleCamSession(2);
            info.cam1calres = this.hlpOrthoSingleCalib(1,info.cam1Sess);
            info.cam2calres = this.hlpOrthoSingleCalib(2,info.cam2Sess);
            info.cam1calresfile = '';
            info.cam2calresfile = '';
            fprintf(1,'Done with single-cam calibrations. Results:\n');
            
            dRP1 = this.hlpAssessOrthoSingleCalib(info.cam1Sess,info.cam1calres.pOpt);
            dRP2 = this.hlpAssessOrthoSingleCalib(info.cam2Sess,info.cam2calres.pOpt);
            mu1 = mean(dRP1(:));
            mu2 = mean(dRP2(:));
            hFig = figure('Name','OrthoCam: Reprojection Error');
            ax = subplot(1,2,1);
            OrthoCam.vizRPerr(ax,dRP1);
            cprms1 = info.cam1Sess.CameraParameters;
            tstr = sprintf('cam1 mono-calib. %dpats, %dpts. mean RPerr=%.3f px',...
              cprms1.NumPatterns,size(cprms1.WorldPoints,1),mu1);
            title(tstr,'fontweight','bold');
            ylabel('count','fontweight','bold');
            ax = subplot(1,2,2);
            OrthoCam.vizRPerr(ax,dRP2);
            cprms2 = info.cam2Sess.CameraParameters;
            tstr = sprintf('cam2 mono-calib. %dpats, %dpts. mean RPerr=%.3f px',...
              cprms2.NumPatterns,size(cprms2.WorldPoints,1),mu2);
            title(tstr,'fontweight','bold');
            
            % summarize intrinsics; and RP err
            tblIntsMono = [ ...
              OrthoCam.summarizeIntrinsics(info.cam1calres.pOpt,info.cam1calres.nPat); ...
              OrthoCam.summarizeIntrinsics(info.cam2calres.pOpt,info.cam2calres.nPat) ];
            tblIntsMono.Properties.RowNames = {'monocal-cam1' 'monocal-cam2'};
            disp(tblIntsMono);
            input('hit enter to continue');
            
            % offer to save monocal res if we just ran
            resp = questdlg('Save single-camera OrthoCam calibrations?',...
              'Save calibrations',...
              'Yes, save','No','Cancel','Yes, save');
            if isempty(resp)
              resp = 'Cancel';
            end
            switch resp
              case 'Yes, save'
                info.cam1calresfile = this.hlpSaveSingleOrthoCamCalRes(info.cam1calres,info.cam1Sess.Filename,1);
                info.cam2calresfile = this.hlpSaveSingleOrthoCamCalRes(info.cam2calres,info.cam2Sess.Filename,2);
                % files may be '' if user cancels
              case 'No'
                % none
              case 'Cancel'
                error('Session:cal','Calibration canceled.');
            end
          case 'loaded'
            fprintf(1,'Loaded single-cam calibrations:\n');
            tblIntsMono = [ ...
              OrthoCam.summarizeIntrinsics(info.cam1calres.pOpt,info.cam1calres.nPat); ...
              OrthoCam.summarizeIntrinsics(info.cam2calres.pOpt,info.cam2calres.nPat) ];
            tblIntsMono.Properties.RowNames = {'monocal-cam1' 'monocal-cam2'};
            disp(tblIntsMono);
            input('hit enter to continue');
        end
        
        % SELECT WORLD COORDSYS/"first" pattern 
%         fpnsStro1 = bset.FullPathNames(1,:)';
        [sel,ok] = listdlg(...
          'PromptString','Select pattern to serve as common World Coordsys (for calibration/optimization only)',...
          'ListString',this.BoardSet.BoardLabels',...
          'SelectionMode','single',...
          'ListSize',[400 400]);
        if ~ok
          error('Session:cal','Calibration canceled.');
        end
        
        boardPerm = [sel 1:sel-1 sel+1:this.BoardSet.NumBoards];
        boardSetUse = struct();
        boardSetUse.FullPathNames = this.BoardSet.FullPathNames(:,boardPerm);
        boardSetUse.BoardPoints = this.BoardSet.BoardPoints(:,:,boardPerm,:);
        boardSetUse.WorldPoints = this.BoardSet.WorldPoints;
        boardSetUse.NumBoards = this.BoardSet.NumBoards;
               
        % pick out extrinsics
        [r2vecs1,t2vecs1] = this.hlpSelSingleCamExtrinsics(1,info.cam1calres,boardSetUse);
        [r2vecs2,t2vecs2] = this.hlpSelSingleCamExtrinsics(2,info.cam2calres,boardSetUse);

        [r2veccam1,t2veccam1,r2veccam2,t2veccam2,rvecsPat,tvecsPat] = ...
          OrthoCam.estimateStroExtrinsics(r2vecs1',t2vecs1',r2vecs2',t2vecs2');
        
        tblInts1 = tblIntsMono(1,:);
        tblInts2 = tblIntsMono(2,:);
        p0 = OrthoCam.packParamsStro( ...
          tblInts1.mx,tblInts1.my,tblInts1.u0,tblInts1.v0,tblInts1.k1,tblInts1.k2,...
          tblInts2.mx,tblInts2.my,tblInts2.u0,tblInts2.v0,tblInts2.k1,tblInts2.k2,...
          r2veccam1,t2veccam1,r2veccam2,t2veccam2,rvecsPat,tvecsPat);
        
        bset = boardSetUse;
        nPat = bset.NumBoards;
        npts = size(bset.BoardPoints,1);

        pOpt = p0;
        while 1
          [pOpt,oFcn] = OrthoCam.calibrateStro(nPat,bset.WorldPoints,...
            bset.BoardPoints(:,:,:,1),bset.BoardPoints(:,:,:,2),pOpt);
          STOP = 'Stop optimization, looks good';
          RESTART = 'Restart optimization';
          CANCEL = 'Cancel';
          resp = questdlg('Restart optimization?','Optimization waypoint',...
            STOP,RESTART,CANCEL,RESTART);
          if isempty(resp)
            resp = CANCEL;
          end
          switch resp
            case STOP
              break;
            case RESTART
              % none; while loop will proceed
            case CANCEL
              error('Session:cal','Calibration canceled.');
          end
        end
              
%         % RP err
%         dRP = oFcn(pOpt);
%         dRP = reshape(dRP,[npts nPat 2]);
%         hFig = figure('Name','OrthoCam: Reprojection Error');
%         dRP1 = dRP(:,:,1);
%         dRP2 = dRP(:,:,2);
%         mu1 = mean(dRP1(:));
%         mu2 = mean(dRP2(:));
%         ax = subplot(1,2,1);
%         OrthoCam.vizRPerr(ax,dRP1);
%         title(sprintf('Stereo calib, cam1. %dpats, %dpts. mean RPerr=%.3f px',nPat,npts,mu1),...
%           'fontweight','bold');
%         ylabel('count','fontweight','bold');
%         ax = subplot(1,2,2);
%         OrthoCam.vizRPerr(ax,dRP2);
%         title(sprintf('Stereo calib, cam2. meanRP err=%.3f px',mu2),'fontweight','bold');

        % Summarize
        tblIntsStro = OrthoCam.summarizeIntrinsicsStro(pOpt,nPat);
        tblInts = [tblIntsMono;tblIntsStro];
        tblInts.Properties.RowNames = ...
          {'cam1/monocal' 'cam2/monocal' 'cam1/strocal' 'cam2/strocal'};
        tblInts = tblInts([1 3 2 4],:);
        disp(tblInts);

        patPtsXYZ = bset.WorldPoints';
        patPtsXYZ = [patPtsXYZ; zeros(1,npts)];

        calObj = OrthoCamCalPair(pOpt,nPat,npts,patPtsXYZ,...
          permute(bset.BoardPoints,[2 1 3 4]),bset.FullPathNames');
        calObj.xformWorldSys(calObj.ijkCamWorld1');
        
        calObj.viewExtrinsics();
        
        calObj.viewRPerr();
        
        if ~isempty(this.Filename)
          [stropath,strofile] = fileparts(this.Filename);
          calResFileProposed = fullfile(stropath,[strofile '_orthocam_strocal.mat']);
        elseif isfield(info,'cam1calresfile') && ~isempty(info.cam1calresfile)
          stropath = fileparts(info.cam1calresfile);
          calResFileProposed = fullfile(stropath,'orthocam_strocal.mat');
        else
          stropath = pwd;
          calResFileProposed = fullfile(stropath,'orthocam_strocal.mat');
        end
        [fname,pth] = uiputfile('*.mat','Save stereo Orthocam calibration.',calResFileProposed);
        if isequal(fname,0)
          % none
        else          
          calResFile = fullfile(pth,fname);
          save(calResFile,'calObj');
          fprintf(1,'Saved %s.\n',calResFile);
        end
        
        msgbox('Stereo Orthocam calibration complete.');
        imagesUsed = [];
        return;                
      end
        
        %------------------------------------------------------------------
        function imagesUsed = calibrateOrig(this)
            if isempty(this.OptimizationOptions) || ...
                    isempty(this.OptimizationOptions.InitialDistortion)
                numRadial = this.CameraModel.NumDistortionCoefficients;                
            else
                numRadial = numel(this.OptimizationOptions.InitialDistortion);
            end
            
            if ~isempty(this.OptimizationOptions)
                initIntrinsics = this.OptimizationOptions.InitialIntrinsics;
                initDistortion = this.OptimizationOptions.InitialDistortion;
            else
                initIntrinsics = [];
                initDistortion = [];
            end
            
            [cameraParams, imagesUsed, estimationErrors] = ...
                estimateCameraParameters(this.BoardSet.BoardPoints, ...
                this.BoardSet.WorldPoints,...
                'EstimateSkew', this.CameraModel.ComputeSkew, ...
                'EstimateTangentialDistortion', ...
                this.CameraModel.ComputeTangentialDistortion, ...
                'NumRadialDistortionCoefficients', numRadial, ...
                'WorldUnits', this.BoardSet.Units, ...
                'ShowProgressBar', true, ...
                'InitialIntrinsicMatrix', initIntrinsics, ...
                'InitialRadialDistortion', initDistortion);
            
            this.CanExport = true;
            this.IsChanged = true;
            this.CameraParameters = cameraParams;
            this.EstimationErrors = estimationErrors;
        end
 
        
        %------------------------------------------------------------------
        function codeString = generateCode(this)
            if isa(this.CameraParameters, 'stereoParameters')
                codeString = generateCodeStereo(this);
            else
                codeString = generateCodeSingle(this);
            end
        end
    end
    
    methods(Access=private)
        %------------------------------------------------------------------
        function codeString = generateCodeSingle(this)
            cameraModel = this.CameraModel;
            codeGenerator = vision.internal.calibration.tool.MCodeGenerator;            

            % Write a header
            codeGenerator.addHeader('cameraCalibrator');

            % Detect checkerboards
            codeGenerator.addComment('Define images to process'); 
            codeGenerator.addLine(sprintf('imageFileNames = %s;',...
                cell2str(this.BoardSet.FullPathNames)));

            codeGenerator.addComment('Detect checkerboards in images');
            codeGenerator.addLine('[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imageFileNames);');
            codeGenerator.addLine('imageFileNames = imageFileNames(imagesUsed);');

            % Set up data for the calibration
            codeGenerator.addComment('Generate world coordinates of the corners of the squares');
            codeGenerator.addLine(sprintf('squareSize = %d;  %% in units of ''%s''',...
                this.BoardSet.SquareSize,...
                this.BoardSet.Units));
            codeGenerator.addLine('worldPoints = generateCheckerboardPoints(boardSize, squareSize);');

            % Calibrate
            if isempty(this.OptimizationOptions)
                initIntrinsics = [];
                initDistortion = [];
            else
                initIntrinsics = this.OptimizationOptions.InitialIntrinsics;
                initDistortion = this.OptimizationOptions.InitialDistortion;
            end
            
            codeGenerator.addComment('Calibrate the camera');
            codeGenerator.addLine(sprintf(['[cameraParams, imagesUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...\n',...
                '''EstimateSkew'', %s, ''EstimateTangentialDistortion'', %s, ...\n', ...
                '''NumRadialDistortionCoefficients'', %d, ''WorldUnits'', ''%s'', ...\n', ...
                '''InitialIntrinsicMatrix'', %s, ''InitialRadialDistortion'', %s);'], ...
                mat2str(cameraModel.ComputeSkew), ...
                mat2str(cameraModel.ComputeTangentialDistortion),...
                cameraModel.NumDistortionCoefficients, this.BoardSet.Units, ...
                mat2str(initIntrinsics), mat2str(initDistortion)));

            % Add visualizations
            
            % Reprojection errors
            codeGenerator.addComment('View reprojection errors');            
            codeGenerator.addLine('h1=figure; showReprojectionErrors(cameraParams);');
            
            % Extrinsics
            codeGenerator.addComment('Visualize pattern locations');
            codeGenerator.addLine(sprintf('h2=figure; showExtrinsics(cameraParams, ''%s'');', ...
                this.ExtrinsicsView));

            % Estimation errors
            codeGenerator.addComment('Display parameter estimation errors');
            codeGenerator.addLine('displayErrors(estimationErrors, cameraParams);');

            % Suggest possible next steps
            codeGenerator.addComment('For example, you can use the calibration data to remove effects of lens distortion.');
            codeGenerator.addLine('originalImage = imread(imageFileNames{1});');
            codeGenerator.addLine('undistortedImage = undistortImage(originalImage, cameraParams);');
            
            codeGenerator.addComment('See additional examples of how to use the calibration data.  At the prompt type:');
            codeGenerator.addLine('% showdemo(''MeasuringPlanarObjectsExample'')');
            codeGenerator.addLine('% showdemo(''StructureFromMotionExample'')');
            
            % Terminate the file with carriage return
            codeGenerator.addReturn();
            
            codeString = codeGenerator.CodeString;            
        end
        
        %------------------------------------------------------------------
        function codeString = generateCodeStereo(this)
            cameraModel = this.CameraModel;
            codeGenerator = vision.internal.calibration.tool.MCodeGenerator;    
            
            % Write a header
            codeGenerator.addHeader('stereoCalibrator');

            % Detect checkerboards
            codeGenerator.addComment('Define images to process'); 
            codeGenerator.addLine(sprintf('imageFileNames1 = %s;',...
                cell2str(this.BoardSet.FullPathNames(1, :))));
            codeGenerator.addLine(sprintf('imageFileNames2 = %s;',...
                cell2str(this.BoardSet.FullPathNames(2, :))));

            codeGenerator.addComment('Detect checkerboards in images');
            codeGenerator.addLine('[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imageFileNames1, imageFileNames2);');

            % Set up data for the calibration
            codeGenerator.addComment('Generate world coordinates of the checkerboard keypoints');
            codeGenerator.addLine(sprintf('squareSize = %d;  %% in units of ''%s''',...
                this.BoardSet.SquareSize,...
                this.BoardSet.Units));
            codeGenerator.addLine('worldPoints = generateCheckerboardPoints(boardSize, squareSize);');

            % Calibrate
            if isempty(this.OptimizationOptions)
                initIntrinsics = [];
                initDistortion = [];
            else
                initIntrinsics = this.OptimizationOptions.InitialIntrinsics;
                initDistortion = this.OptimizationOptions.InitialDistortion;
            end
            
            codeGenerator.addComment('Calibrate the camera');
            codeGenerator.addLine(sprintf(['[stereoParams, pairsUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...\n',...
                '''EstimateSkew'', %s, ''EstimateTangentialDistortion'', %s, ...\n', ...
                '''NumRadialDistortionCoefficients'', %d, ''WorldUnits'', ''%s'', ...\n', ...
                '''InitialIntrinsicMatrix'', %s, ''InitialRadialDistortion'', %s);'], ...
                mat2str(cameraModel.ComputeSkew), ...
                mat2str(cameraModel.ComputeTangentialDistortion),...
                cameraModel.NumDistortionCoefficients, this.BoardSet.Units, ...
                mat2str(initIntrinsics), mat2str(initDistortion)));

            % Add visualizations
            
            % Reprojection errors
            codeGenerator.addComment('View reprojection errors');            
            codeGenerator.addLine('h1=figure; showReprojectionErrors(stereoParams);');
            
            % Extrinsics
            codeGenerator.addComment('Visualize pattern locations');
            codeGenerator.addLine(sprintf('h2=figure; showExtrinsics(stereoParams, ''%s'');', ...
                this.ExtrinsicsView));
            
            % Estimation errors
            codeGenerator.addComment('Display parameter estimation errors');
            codeGenerator.addLine('displayErrors(estimationErrors, stereoParams);');
            
            % Suggest possible next steps
            codeGenerator.addComment('You can use the calibration data to rectify stereo images.');
            codeGenerator.addLine('I1 = imread(imageFileNames1{1});');
            codeGenerator.addLine('I2 = imread(imageFileNames2{1});');
            codeGenerator.addLine('[J1, J2] = rectifyStereoImages(I1, I2, stereoParams);');
            
            codeGenerator.addComment('See additional examples of how to use the calibration data.  At the prompt type:');
            codeGenerator.addLine('% showdemo(''StereoCalibrationAndSceneReconstructionExample'')');
            codeGenerator.addLine('% showdemo(''DepthEstimationFromStereoVideoExample'')');
            
            % Terminate the file with carriage return
            codeGenerator.addReturn();
            
            codeString = codeGenerator.CodeString;
        end
    end
    
    %----------------------------------------------------------------------
    % saveobj and loadobj are implemented to ensure compatibility across
    % releases even if architecture of Session class changes
    methods (Hidden)
       
        function that = saveobj(this)            
            that.version         = this.Version;
            that.cameraModel     = this.CameraModel;
            that.optimizationOptions = this.OptimizationOptions;
            that.cameraParams    = this.CameraParameters;
            that.estimationErrors = this.EstimationErrors;
            that.shouldExportErrors = this.ShouldExportErrors;
            that.boardSet        = this.BoardSet;
            that.hasEnoughBoards = this.HasEnoughBoards;
            that.canExport       = this.CanExport;
            that.isChanged       = this.IsChanged;
            that.extrinsicsView  = this.ExtrinsicsView;
            that.errorsView      = this.ErrorsView;
            that.filename        = this.FileName;
            that.exportVarName   = this.ExportVariableName;
        end
        
    end
    
    %----------------------------------------------------------------------
    methods (Static, Hidden)
       
        function this = loadobj(that)
            if isa(that, 'vision.internal.calibration.tool.Session')
                this = that;
                this.OptimizationOptions.InitialIntrinsics = [];
                this.OptimizationOptions.InitialDistortion = [];
            else
                this = vision.internal.calibration.tool.Session;
                this.CameraModel        = that.cameraModel;
                this.CameraParameters   = that.cameraParams;
                this.EstimationErrors   = that.estimationErrors;
                this.ShouldExportErrors = that.shouldExportErrors;
                this.BoardSet           = that.boardSet;
                this.HasEnoughBoards    = that.hasEnoughBoards;
                this.CanExport          = that.canExport;
                this.IsChanged          = that.isChanged;
                this.ExtrinsicsView     = that.extrinsicsView;
                this.FileName           = that.filename;
                this.ExportVariableName = that.exportVarName;
                
                if isfield(that, 'optimizationOptions')
                    this.OptimizationOptions = that.optimizationOptions;
                else
                    this.OptimizationOptions.InitialIntrinsics = [];
                    this.OptimizationOptions.InitialDistortion = [];
                end
            end
        end
        
    end
end

%--------------------------------------------------------------
% This function handles conversion of cell array of strings
% to a string representing the entire cell array
%--------------------------------------------------------------
function str = cell2str(cellArray)
str = '{'; % opening bracket

% constants that are easier to read once assigned into
% variables
quote = '''';
nextLine = sprintf(',...\n');

for i=1:numel(cellArray)
    str = [str, quote, cellArray{i}, quote, nextLine]; %#ok<AGROW>
end

str = [str, '}']; % closing bracket
end
