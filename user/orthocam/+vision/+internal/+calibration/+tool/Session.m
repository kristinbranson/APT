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
        function imagesUsed = calibrate(this)
            disp('MINE');
            pause(3);
          
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
