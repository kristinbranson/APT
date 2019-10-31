classdef stereoParameters < vision.internal.calibration.StereoParametersImpl & matlab.mixin.CustomDisplay 
    % stereoParameters Object for storing parameters of a stereo camera system
    %
    %   stereoParams = stereoParameters(cameraParameters1, ...
    %     cameraParameters2, rotationOfCamera2, translationOfCamera2) returns an
    %     object that contains parameters of stereo camera system.
    %     cameraParameters1 and cameraParameters2 are
    %     cameraParameters objects containing intrinsics and extrinsics
    %     of camera 1 and camera 2 respectively. rotationOfCamera2 is a
    %     3-by-3 matrix representing the rotation of camera 2 relative to
    %     camera 1. translationOfCamera2 is a 3-element vector representing the
    %     translation of camera 2 relative camera 1.
    %
    %   stereoParams = stereoParameters(paramStruct) returns a
    %     stereoParameters object containing the parameters specified by
    %     paramStruct returned by the toStruct method. 
    %
    %  stereoParameters properties:
    %
    %      Intrinsic and extrinsic parameters of the two cameras:
    %      ------------------------------------------------------
    %      CameraParameters1 - Parameters of the camera 1
    %      CameraParameters2 - Parameters of the camera 2
    %
    %      Geometric relationship between the two cameras:
    %      -----------------------------------------------
    %      RotationOfCamera2    - Rotation of camera 2 relative to camera 1
    %      TranslationOfCamera2 - Translation of camera 2 relative to camera 1
    %      FundamentalMatrix    - The fundamental matrix of the stereo system
    %      EssentialMatrix      - The essential matrix of the stereo system
    %
    %      Accuracy of estimated parameters:
    %      ---------------------------------
    %      MeanReprojectionError - Average reprojection error in pixels
    %
    %      Settings used to estimate camera parameters:
    %      --------------------------------------------
    %      NumPatterns          - Number of patterns used to estimate extrinsics
    %      WorldPoints          - World coordinates of pattern keypoints
    %      WorldUnits           - Units of the world coordinates
    %
    %  stereoParameters methods:
    %      toStruct - convert a stereoParameters object into a struct
    %
    % Example:
    % -----------------------------
    % % Specify calibration images
    % imageDir = fullfile(toolboxdir('vision'), 'visiondata', ...
    %     'calibration', 'stereo');
    % leftImages = imageDatastore(fullfile(imageDir, 'left'));
    % rightImages = imageDatastore(fullfile(imageDir, 'right'));
    %
    % % Detect the checkerboards
    % [imagePoints, boardSize] = detectCheckerboardPoints(...
    %     leftImages.Files, rightImages.Files);
    %
    % % Specify world coordinates of checkerboard keypoints
    % squareSize = 108; % in millimeters
    % worldPoints = generateCheckerboardPoints(boardSize, squareSize);
    %
    % % Calibrate the stereo camera system
    % params = estimateCameraParameters(imagePoints, worldPoints);
    %
    % % Visualize calibration accuracy
    % showReprojectionErrors(params);
    %
    % See also estimateCameraParameters, stereoCameraCalibrator, 
    %   cameraParameters, showReprojectionErrors, showExtrinsics, 
    %   rectifyStereoImages, reconstructScene
    
    % Copyright 2014 MathWorks, Inc.
        
    methods(Access=private, Static)
        function name = matlabCodegenRedirect(~)
            name = 'vision.internal.calibration.StereoParametersImpl';
        end
    end
    
    methods
        function this = stereoParameters(varargin)                        
            this = this@vision.internal.calibration.StereoParametersImpl(...
                varargin{:});            
        end        
    end
    
    methods(Access=protected)
        %------------------------------------------------------------------
        % Group properties into meaningful categories for display
        %------------------------------------------------------------------
        function group = getPropertyGroups(~)
            group1 = 'Parameters of Two Cameras';
            list1 = {'CameraParameters1', 'CameraParameters2'};
            
            group2 = 'Inter-camera Geometry';
            list2 = {'RotationOfCamera2', 'TranslationOfCamera2', ...
                'FundamentalMatrix', 'EssentialMatrix'};
            
            group3 = 'Accuracy of Estimation';
            list3 = {'MeanReprojectionError'};
            
            group4 = 'Calibration Settings';
            list4 = {'NumPatterns', 'WorldPoints', 'WorldUnits'};
            
            group(1) = matlab.mixin.util.PropertyGroup(list1, group1);
            group(2) = matlab.mixin.util.PropertyGroup(list2, group2);
            group(3) = matlab.mixin.util.PropertyGroup(list3, group3);
            group(4) = matlab.mixin.util.PropertyGroup(list4, group4);
        end
    end
    
    methods(Hidden)       
        %------------------------------------------------------------------
        % This method is invoked by the showReprojectionErrors function,
        % which does all the parameter parsing and validation.
        %------------------------------------------------------------------
        function hAxes = showReprojectionErrorsImpl(this, ~, hAxes, highlightIndex)
            
            hAxes = newplot(hAxes);
            
            [meanError1, meanErrorsPerImage1] = ...
                computeMeanError(this.CameraParameters1);
            [meanError2, meanErrorsPerImage2] = ...
                computeMeanError(this.CameraParameters2);
            allErrors = [meanErrorsPerImage1, meanErrorsPerImage2];
            meanError = mean([meanError1, meanError2]);
            
            % Record the current 'hold' state so that we can restore it later
            holdState = get(hAxes,'NextPlot');
            
            % Plot the errors
            hBar = bar(hAxes, allErrors);
            set(hBar(1), 'FaceColor', [0, 0.7, 1]);
            set(hBar(2), 'FaceColor', [242, 197, 148] / 255);
            set(hBar, 'Tag', 'errorBars');
            
            set(hAxes, 'NextPlot', 'add'); % hold on
            hErrorLine = line(get(hAxes, 'XLim'), [meanError, meanError],...
                'LineStyle', '--', 'Parent', hAxes);
            legend([hBar, hErrorLine], 'Camera 1', 'Camera 2', ...
                getString(message(...
                'vision:calibrate:overallMeanError', ...
                sprintf('%.2f', meanError))), ...
                'Location', 'SouthEast');
            
            % Plot highlighted errors
            highlightedErrors = allErrors;
            highlightedErrors(~highlightIndex, :) = 0;
            hHighlightedBar = bar(hAxes, highlightedErrors);
            set(hHighlightedBar(1), 'FaceColor', [0 0 1]);
            set(hHighlightedBar(2), 'FaceColor', [190, 101, 1] ./ 255);
            set(hHighlightedBar, 'Tag', 'highlightedBars');
            
            set(hAxes, 'NextPlot', holdState); % restore the hold state
            
            title(hAxes, getString(message('vision:calibrate:barGraphTitle')));
            xlabel(hAxes, getString(message('vision:calibrate:barGraphXLabelStereo')));
            ylabel(hAxes, getString(message('vision:calibrate:barGraphYLabel')));
        end
        
        %------------------------------------------------------------------
        function errors = refine(this, imagePointsLeft, imagePointsRight, ...
                shouldComputeErrors)
            x0 = serialize(this);
            numImages = this.CameraParameters1.NumPatterns;
            xdata = repmat(this.CameraParameters2.WorldPoints, [2 * numImages, 1]);
            ydataLeft = arrangeImagePointsIntoMatrix(imagePointsLeft);
            ydataRight = arrangeImagePointsIntoMatrix(imagePointsRight);
            ydata = [ydataLeft; ydataRight];
%            options = optimset('Display', 'off');     
            options = optimset('Display', 'iter', 'TolFun',1e-9,'tolX',1e-9,'maxiter',2e4,'MaxFunEvals',1e6);

            
            worldPoints = this.WorldPoints;
            worldPointsHomog = [worldPoints, ones(size(worldPoints, 1), 1)];
            numPatterns = this.NumPatterns;
            
            if shouldComputeErrors
                [x, ~, residual, ~, ~, ~, jacobian] = ...
                    lscftsh(@reprojectWrapper, x0, xdata, ydata, [], [], options);
            
                standardError = ...
                    vision.internal.calibration.computeStandardError(jacobian, ...
                    residual);
                
                % be careful with memory
                clear jacobian;
                
                errors = deserializeErrors(this, standardError);
            else
                x = ...
                    lscftsh(@reprojectWrapper, x0, xdata, ydata, [], [], options);
                errors = [];
            end
            
            deserialize(this, x);
            
            computeReprojectionErrors(...
                this.CameraParameters1, imagePointsLeft);
            computeReprojectionErrors(...
                this.CameraParameters2, imagePointsRight);
            
            % Resetting LeftSerializedLength to make isequal() behave
            % reasonably. If we do not do that an object returned by
            % estimateCameraParameters and an identical object returned by the
            % constructor will not be equal according to isequal.
            this.LeftSerializedLength = [];
            
            %----------------------------------------------------------------------
            function reprojectedPoints = reprojectWrapper(paramsVector, ~)
                
                [r2, t2, camera1, camera2] = unpackSerializedParams(this, paramsVector);
                camera1 = unpackSerializedParams(this.CameraParameters1, camera1);
                camera2 = unpackSerializedParams(this.CameraParameters2, camera2);
                R2 = vision.internal.calibration.rodriguesVectorToMatrix(r2);
                
                intrinsicMatrix1 = ...
                    vision.internal.calibration.constructIntrinsicMatrix(...
                    camera1.focalLength(1), camera1.focalLength(2), ...
                    camera1.principalPoint(1), camera1.principalPoint(2), ...
                    camera1.skew);
                
                intrinsicMatrix2 = ...
                    vision.internal.calibration.constructIntrinsicMatrix(...
                    camera2.focalLength(1), camera2.focalLength(2), ...
                    camera2.principalPoint(1), camera2.principalPoint(2), ...
                    camera2.skew);
                
                reprojectedPoints1 = zeros([size(worldPoints), numPatterns]);                
                reprojectedPoints2 = zeros([size(worldPoints), numPatterns]);
                
                for i = 1:numPatterns
                    % camera 1
                    R = vision.internal.calibration.rodriguesVectorToMatrix(...
                        camera1.rotationVectors(i, :));
                    t = camera1.translationVectors(i, :)';
                                        
                    reprojPointsHomog = worldPointsHomog * (intrinsicMatrix1 * ...
                        [R(:, 1:2), t])';
                    reprojectedPoints1(:,:,i) = bsxfun(@rdivide, ...
                        reprojPointsHomog(:, 1:2), reprojPointsHomog(:,3));
                
                    % camera 2
                    R = R2 * R;
                    t = t2' + R2 * t;
                    reprojPointsHomog = worldPointsHomog * (intrinsicMatrix2 * ...
                        [R(:, 1:2), t])';
                    reprojectedPoints2(:,:,i) = bsxfun(@rdivide, ...
                        reprojPointsHomog(:, 1:2), reprojPointsHomog(:,3));
                end
                
                % apply distortion
                
                reprojectedPoints1 = visionDistortPoints(reprojectedPoints1, ...
                    intrinsicMatrix1, camera1.radialDistortion, ...
                    camera1.tangentialDistortion);
                
                reprojectedPoints2 = visionDistortPoints(reprojectedPoints2, ...
                    intrinsicMatrix2, camera2.radialDistortion, ...
                    camera2.tangentialDistortion);

                reprojectedPoints = cat(3, reprojectedPoints1, reprojectedPoints2);
                reprojectedPoints = arrangeImagePointsIntoMatrix(reprojectedPoints);
            end
            
            %----------------------------------------------------------------------
            function pointMatrix = arrangeImagePointsIntoMatrix(imagePoints)
                pointMatrix = reshape(permute(imagePoints, [2, 1, 3]), ...
                    [2, size(imagePoints, 1) * size(imagePoints, 3)])';
            end
        end
        
        %------------------------------------------------------------------
        function x = serialize(this)
            rvec = vision.internal.calibration.rodriguesMatrixToVector(...
                this.RotationOfCamera2');
            x = [rvec(:); this.TranslationOfCamera2(:)];
            x1 = serialize(this.CameraParameters1);
            this.LeftSerializedLength = numel(x1);
            x2 = serialize(this.CameraParameters2);
            numExtrinsicElements = 2 * 3 * this.CameraParameters2.NumPatterns;
            x = [x; x1; x2(1:end-numExtrinsicElements)];
        end
        
        %------------------------------------------------------------------
        function deserialize(this, x)
            [r, t, camera1, camera2] = unpackSerializedParams(this, x);
            
            % Rotation of camera 2: 3 elements
            this.RotationOfCamera2 = vision.internal.calibration.rodriguesVectorToMatrix(r)';
            
            % Translation of camera 2: 3 elements
            this.TranslationOfCamera2 = t;
            
            deserialize(this.CameraParameters1, camera1);
            
            % CameraParameters2
            deserialize(this.CameraParameters2, camera2);
            [rvecsR, tvecsR] = this.computeRightExtrinsics;
            this.CameraParameters2.setExtrinsics(rvecsR, tvecsR);
        end         
    end
        
     methods(Access=private)
         %------------------------------------------------------------------
        function [rvecs, tvecs] = computeRightExtrinsics(this)
            numImages = this.CameraParameters1.NumPatterns;
            rvecs = zeros(numImages, 3);
            tvecs = zeros(numImages, 3);
            for i = 1:numImages
                currR = this.RotationOfCamera2' * ...
                    this.CameraParameters1.RotationMatrices(:, :, i)';
                rvecs(i, :) = vision.internal.calibration.rodriguesMatrixToVector(currR);
                tvecs(i, :) = (this.TranslationOfCamera2' + ...
                    this.RotationOfCamera2' * ...
                    this.CameraParameters1.TranslationVectors(i, :)')';
            end
        end
        
        %------------------------------------------------------------------
        function [r, t, camera1, camera2] = unpackSerializedParams(this, x)
            r = x(1:3)';
            t = x(4:6)';
            
            first = 7;
            leftLength = this.LeftSerializedLength;
            last = first + leftLength - 1;
            camera1 = x(first:last);
            camera2 = x(last+1:end);
        end
        
        %------------------------------------------------------------------
        function errors = deserializeErrors(this, x)
            [r, t, camera1, camera2] = unpackSerializedParams(this, x);
            errorStruct.r = r;
            errorStruct.t = t;
            errorStruct.camera1 = unpackSerializedParams(...
                this.CameraParameters1, camera1);
            
            errorStruct.camera2 = unpackSerializedParams(...
                this.CameraParameters2, camera2);
            errors = stereoCalibrationErrors(errorStruct);
        end
     end
     
    %----------------------------------------------------------------------
    % saveobj and loadobj are implemented to ensure compatibility across
    % releases even if architecture of stereoParameters class changes
    methods(Hidden)
        function that = saveobj(this)
            that.CameraParameters1    = this.CameraParameters1;
            that.CameraParameters2    = this.CameraParameters2;
            that.RotationOfCamera2    = this.RotationOfCamera2;
            that.TranslationOfCamera2 = this.TranslationOfCamera2;
            
            that.RectificationParams = this.RectificationParams;
            that.Version = this.Version;
        end
    end
    
    %----------------------------------------------------------------------
    methods (Static, Hidden)
        
        function this = loadobj(that)
            this = stereoParameters(that.CameraParameters1, ...
                that.CameraParameters2, that.RotationOfCamera2, ...
                that.TranslationOfCamera2);
            this.RectificationParams = that.RectificationParams;
        end
    end
end
