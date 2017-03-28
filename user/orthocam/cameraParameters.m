classdef cameraParameters < vision.internal.calibration.CameraParametersImpl & matlab.mixin.CustomDisplay
    %cameraParameters Object for storing camera parameters
    %
    %   cameraParams = cameraParameters returns an object that
    %   contains intrinsic, extrinsic, and lens distortion parameters of a
    %   camera.
    %
    %   cameraParams = cameraParameters(paramStruct) returns a
    %   cameraParameters object containing the parameters specified by
    %   paramStruct returned by the toStruct method. 
    %
    %   cameraParams = cameraParameters(Name, Value, ...)
    %   configures the camera parameters object properties, specified as one or
    %   more name-value pair arguments. Unspecified properties have default
    %   values. The available parameters are:
    %
    %   'IntrinsicMatrix'      A 3-by-3 projection matrix of the form
    %                          [fx 0 0; s fy 0; cx cy 1], where [cx, cy] are the
    %                          coordinates of the optical center (the principal
    %                          point) in pixels and s is the skew parameter which
    %                          is 0 if the x and y axis are exactly perpendicular.
    %                          fx = F * sx and fy = F * sy, where F is the focal
    %                          length in world units, typically millimeters, and
    %                          [sx, sy] are the number of pixels per world unit
    %                          in the x and y direction respectively. Thus, fx
    %                          and fy are in pixels.
    %
    %                          Default: eye(3)
    %
    %   'RadialDistortion'     A 2-element vector [k1 k2] or a 3-element vector
    %                          [k1 k2 k3]. If a 2-element vector is supplied, k3 is
    %                          assumed to be 0. The radial distortion is caused by
    %                          the fact that light rays are bent more the farther
    %                          away they are from the optical center. Distorted
    %                          location of a point is computed as follows:
    %                          x_distorted = x(1 + k1 * r^2 + k2 * r^4 + k3 * r^6)
    %                          y_distorted = y(1 + k1 * r^2 + k2 * r^4 + k3 * r^6)
    %                          where [x,y] is a non-distorted image point in
    %                          normalized image coordinates in world units with the
    %                          origin at the optical center, and r^2 = x^2 + y^2.
    %                          Typically, two coefficients are sufficient, and k3 is
    %                          only needed for wide-angle lenses.
    %
    %                          Default: [0 0 0]
    %
    %   'TangentialDistortion' A 2-element vector [p1 p2]. Tangential distortion
    %                          is caused by the lens not being exactly parallel to
    %                          to the image plane. Distorted location of a point
    %                          is computed as follows:
    %                          x_distorted = x + [2 * p1 * y + p2 * (r^2 + 2 * x^2)]
    %                          y_distorted = y + [p1 * (r^2 + 2*y^2) + 2 * p2 * x]
    %                          where [x,y] is a non-distorted image point in
    %                          normalized image coordinates in world units with
    %                          the origin at the optical center, and r^2 = x^2 + y^2.
    %
    %                          Default: [0 0]
    %
    %   'RotationVectors'      An M-by-3 matrix containing M rotation vectors
    %                          Each vector describes the 3-D rotation of the
    %                          camera's image plane relative to the corresponding
    %                          calibration pattern. The vector specifies the
    %                          3-D axis about which the camera is rotated, and
    %                          its magnitude is the rotation angle in radians.
    %
    %                          Default: []
    %
    %   'TranslationVectors'   An M-by-3 matrix containing M translation vectors.
    %                          Each vector describes the translation of the
    %                          camera's image plane relative to the corresponding
    %                          calibration pattern in world units.
    %
    %                          Default: []
    %
    %   'WorldPoints'          An M-by-2 array of [x,y] world coordinates of
    %                          keypoints on the calibration pattern, where M is
    %                          the number of keypoints in the pattern.
    %                          WorldPoints must be non-empty for showExtrinsics
    %                          to work.
    %
    %                          Default: []
    %
    %   'WorldUnits'           A string describing the units, in which the
    %                          WorldPoints are specified.
    %
    %                          Default: 'mm'
    %
    %   'EstimateSkew'         A logical scalar that specifies whether image axes
    %                          skew was estimated. When set to false, the image
    %                          axes are assumed to be exactly perpendicular.
    %
    %                          Default: false
    %
    %   'NumRadialDistortionCoefficients'  2 or 3. Specifies the number of radial
    %                                      distortion coefficients that were
    %                                      estimated.
    %
    %                                      Default: 2
    %
    %   'EstimateTangentialDistortion'     A logical scalar that specifies
    %                                      whether tangential distortion was
    %                                      estimated. When set to false, tangential
    %                                      distortion is assumed to be negligible.
    %
    %                                      Default: false
    %
    %   'ReprojectionErrors'  An M-by-2-by-P array of [x,y] pairs representing
    %                         the translation in x and y between the reprojected 
    %                         pattern keypoints and the detected pattern keypoints.             
    %
    %
    %   cameraParameters properties:
    %
    %      Intrinsic camera parameters:
    %      ----------------------------
    %      IntrinsicMatrix      - 3-by-3 projection matrix
    %      PrincipalPoint       - Coordinates of the camera's optical center
    %      FocalLength          - Focal length in x and y in pixels
    %      Skew                 - Camera axes skew
    %
    %      Camera lens distortion:
    %      -----------------------
    %      RadialDistortion     - Caused by curvature of the lens
    %      TangentialDistortion - Caused by lens not being parallel to sensor
    %
    %      Extrinsic camera parameters:
    %      ----------------------------
    %      RotationMatrices     - Rotations of the camera in matrix form
    %      RotationVectors      - Rotations of the camera in axis-angle form
    %      TranslationVectors   - Translations of the camera
    %
    %      Accuracy of estimated camera parameters:
    %      ----------------------------------------
    %      MeanReprojectionError - Average reprojection error over all patterns
    %      ReprojectionErrors    - Translation between projected and detected points
    %      ReprojectedPoints     - World points reprojected onto calibration images
    %
    %      Settings used to estimate camera parameters:
    %      --------------------------------------------
    %      NumPatterns                     - Number of patterns used to estimate extrinsics
    %      WorldPoints                     - World coordinates of pattern keypoints
    %      WorldUnits                      - Units of the world coordinates
    %      EstimateSkew                    - True if axis skew was estimated
    %      NumRadialDistortionCoefficients - Number of radial distortion coefficients
    %      EstimateTangentialDistortion    - True if tangential distortion was estimated
    %
    %   cameraParameters methods:
    %      pointsToWorld - Map image points onto X-Y plane in world coordinates
    %      worldToImage  - Project world points into the image
    %      toStruct      - Convert a cameraParameters object into a struct
    %
    %   Notes:
    %   ------
    %   RotationVectors and TranslationVectors must be set together in the
    %   constructor to ensure that the number of translation and rotation
    %   vectors is the same. Setting one but not the other will result in an
    %   error.
    %
    %   Example
    %   -------
    %    % Create a cameraParameters object manually.
    %    % In practice use estimateCameraParameters or cameraCalibrator app.
    %    IntrinsicMatrix = [715.2699   0       0;
    %                          0     711.5281  0;
    %                      565.6995  355.3466  1];
    %    radialDistortion = [-0.3361 0.0921];
    %    cameraParams = cameraParameters('IntrinsicMatrix', IntrinsicMatrix, ...
    %        'RadialDistortion', radialDistortion)
    %
    %   See also estimateCameraParameters, cameraCalibrator, undistortImage,
    %            showExtrinsics, showReprojectionErrors, stereoParameters
    
    methods(Access=private, Static)
       function name = matlabCodegenRedirect(~)
         name = 'vision.internal.calibration.CameraParametersImpl';
       end
    end
   
    methods(Access=protected)
        %------------------------------------------------------------------
        % Group properties into meaningful categories for display
        %------------------------------------------------------------------
        function group = getPropertyGroups(~)
            group1 = 'Camera Intrinsics';
            list1 = {'IntrinsicMatrix', 'FocalLength', 'PrincipalPoint', 'Skew'};
            
            group2 = 'Lens Distortion';
            list2 = {'RadialDistortion', 'TangentialDistortion'};
            
            group3 = 'Camera Extrinsics';
            list3 = {'RotationMatrices', 'TranslationVectors'};
            
            group4 = 'Accuracy of Estimation';
            list4 = {'MeanReprojectionError', 'ReprojectionErrors', ...
                'ReprojectedPoints'};
            
            group5 = 'Calibration Settings';
            list5 = {'NumPatterns', 'WorldPoints', 'WorldUnits', ...
                'EstimateSkew', 'NumRadialDistortionCoefficients', ...
                'EstimateTangentialDistortion'};
            
            group(1) = matlab.mixin.util.PropertyGroup(list1, group1);
            group(2) = matlab.mixin.util.PropertyGroup(list2, group2);
            group(3) = matlab.mixin.util.PropertyGroup(list3, group3);
            group(4) = matlab.mixin.util.PropertyGroup(list4, group4);
            group(5) = matlab.mixin.util.PropertyGroup(list5, group5);
        end
    end
    
    methods
        %----------------------------------------------------------------------
        function this = cameraParameters(varargin)                         
            this@vision.internal.calibration.CameraParametersImpl(varargin{:});                        
        end               
    end
        
    methods(Hidden)
        %------------------------------------------------------------------
        function undistortedPoints = undistortPointsImpl(this, points)
            options = optimset('Display', 'off');
            undistortedPoints = ...
                lscftsh(@this.distortPoints, points, [], points, [], [], options);
        end
    end
    
    methods(Hidden)
        %------------------------------------------------------------------        
        function hAxes = showReprojectionErrorsImpl(this, view, hAxes, highlightIndex)
            % showReprojectionErrors Visualize calibration errors.
            %   showReprojectionErrors(cameraParams) displays a bar 
            %   graph that represents the accuracy of camera calibration. 
            %   The bar graph displays the mean reprojection error per image. 
            %   The cameraParams input is returned from the 
            %   estimateCameraParameters function or from the Camera 
            %   Calibrator app. 
            % 
            %   showReprojectionErrors(cameraParams, view) displays the 
            %   errors using the visualization style specified by the view 
            %   input. 
            %   Valid values of view:
            %   'BarGraph':    Displays mean error per image as a bar graph.
            %
            %   'ScatterPlot': Displays the error for each point as a scatter plot.
            %
            %   ax = showReprojectionErrors(...) returns the plot's axes handle.
            %
            %   showReprojectionErrors(...,Name,Value) specifies additional 
            %   name-value pair arguments described below:
            %
            %   'HighlightIndex' Indices of selected images, specified as a 
            %   vector of integers. For the 'BarGraph' view, bars corresponding 
            %   to the selected images are highlighted. For 'ScatterPlot' view, 
            %   points corresponding to the selected images are displayed with 
            %   circle markers.
            %
            %   Default: []
            %
            %   'Parent'         Axes for displaying plot.
            %
            %   Class Support
            %   -------------
            %   cameraParams must be a cameraParameters object.
            %
            
            if isempty(this.ReprojectionErrors)
                error(message('vision:calibrate:cannotShowEmptyErrors'));
            end
            hAxes = newplot(hAxes);
            
            errors = this.ReprojectionErrors;      
                            
            if strcmpi(view, 'bargraph')
                plotMeanErrorPerImage(hAxes, highlightIndex);
            else % 'scatterPlot'
                plotAllErrors(hAxes, errors, highlightIndex);
            end

            %--------------------------------------------------------------
            % Plot a bar graph with error bars
            function plotMeanErrorPerImage(hAxes, highlightIndex)
                % Record the current 'hold' state so that we can restore it later
                holdState = get(hAxes,'NextPlot');
                                
                % compute mean errors per image
                [meanError, meanErrors] = computeMeanError(this);
                                
                % plot the mean errors
                hBar = bar(hAxes, meanErrors, 'FaceColor', [0 0.7 1]);                
                set(hBar, 'Tag', 'errorBars');
                
                set(hAxes, 'NextPlot', 'add'); % hold on     
                
                % plot errors for highlighted images
                highlightedErrors = meanErrors;
                highlightedErrors(~highlightIndex) = 0;
                hHighlightedBar = bar(hAxes, highlightedErrors, ...
                    'FaceColor', [0 0 1]);
                set(hHighlightedBar, 'Tag', 'highlightedBars');
                
                hErrorLine = line(get(hAxes, 'XLim'), [meanError, meanError],...
                    'LineStyle', '--', 'Parent', hAxes);
                legend(hErrorLine, getString(message(...
                    'vision:calibrate:overallMeanError', ...
                    sprintf('%.2f', meanError))), ...
                    'Location', 'SouthEast');
 
                set(hAxes, 'NextPlot', holdState); % restore the hold state                                
                
                title(hAxes, getString(message('vision:calibrate:barGraphTitle')));
                xlabel(hAxes, getString(message('vision:calibrate:barGraphXLabel')));
                ylabel(hAxes, getString(message('vision:calibrate:barGraphYLabel')));
            end            
            
            %--------------------------------------------------------------
            % Plot a scatter plot of X vs. Y
            function plotAllErrors(hAxes, errors, highlightIndex)
                % Record the current 'hold' state so that we can restore it later
                holdState = get(hAxes,'NextPlot');
                
                % colormap for marker colors
                colorLookup = im2double(label2rgb(1:this.NumPatterns, ...
                    'lines','c','shuffle'));
                
                % plot the errors
                legendStrings = cell(1, this.NumPatterns);
                for i = 1:this.NumPatterns
                    legendStrings{i} = sprintf('%d', i);
                    x = errors(:, 1, i);
                    y = errors(:, 2, i);
                    if highlightIndex(i)
                        marker = 'o';
                    else
                        marker = '+';
                    end
                    color = squeeze(colorLookup(1,i,:))';
                    plot(hAxes, x, y, marker, 'MarkerEdgeColor', color);                    
                    set(hAxes, 'NextPlot', 'add'); % hold on 
                end
                
                drawnow();
                % plot highlighted points again to make them more visible
                for i = 1:this.NumPatterns
                    if highlightIndex(i)
                        x = errors(:, 1, i);
                        y = errors(:, 2, i);
                        marker = 'o';
                        color = squeeze(colorLookup(1,i,:))';
                        plot(hAxes, x, y, marker, 'MarkerEdgeColor', color);   
                    end
                end
                
                legend(hAxes, legendStrings);
                title(hAxes, getString(message('vision:calibrate:scatterPlotTitle')));
                xlabel(hAxes, getString(message('vision:calibrate:scatterPlotXLabel')));
                ylabel(hAxes, getString(message('vision:calibrate:scatterPlotYLabel')));

                axis equal;
                set(hAxes, 'NextPlot', holdState); % restore the hold state                 
            end
        end        
    end
    

    
    methods (Hidden=true, Access=public)
        function errors = refine(this, imagePoints, shouldComputeErrors)
            % refine Estimate camera parameters numerically.
            %
            % params = refine(this, imagePoints) 
            %   numerically refines an initial guess of camera parameter values.
            %   accounting for lens distortion.
            %
            % this is a vision.CameraParameters object containing the 
            % initial "guess" of the parameter values. These initial values
            % can be computed in closed form by assuming zero lens distortion.
            %
            % imagePoints is an M x 2 x P array containing the [x y] 
            % coordinates of the points detected in images, in pixels. M is
            % the number of points in the pattern and P is the number of images.

            x0 = serialize(this);            
            numImages = size(this.RotationVectors, 1);
            xdata = repmat(this.WorldPoints, [numImages, 1]);            
            ydata = arrangeImagePointsIntoMatrix(imagePoints);    
               
            %options = optimset('Display', 'iter', 'Jacobian', 'on','TolFun',1e-8,'tolX',1e-10,'maxiter',2e4);
            options = optimset('Display', 'iter', 'Jacobian', 'on','TolFun',1e-8,'tolX',1e-10,'maxiter',2);
            
            worldPoints = this.WorldPoints;
            worldPointsXYZ = [worldPoints, zeros(size(worldPoints, 1), 1)]';
            numWorldPoints = size(worldPointsXYZ, 2);
            numPatterns = this.NumPatterns;
                        
            if shouldComputeErrors
                [x, ~, residual, ~, ~, ~, jacobian] = ...
                    lscftsh(@reprojectWrapper, x0, xdata, ydata, [], [], options);

                standardError = ...
                    vision.internal.calibration.computeStandardError(jacobian, ...
                    residual);
                
                % be careful with memory
                clear jacobian;
                
                errorsStruct = unpackSerializedParams(this, standardError);
                errors = cameraCalibrationErrors(errorsStruct);
            else
                x = ...
                    lscftsh(@reprojectWrapper, x0, xdata, ydata, [], [], options);
                errors = [];
            end
            
            deserialize(this, x);
            computeReprojectionErrors(this, imagePoints);

            %----------------------------------------------------------------------
            function [reprojectedPoints, jacobian] = reprojectWrapper(paramsVector, ~)
                paramStruct = unpackSerializedParams(this, paramsVector);
                reprojectedPoints = zeros([size(worldPoints), numPatterns]);
                jacobian = zeros(numel(reprojectedPoints), numel(x0));
                numRadialDistortionCoeffs = numel(paramStruct.radialDistortion);
                
                for i = 1:numPatterns
                    % x = [fx; fy; cx; cy; skew; radial; tangential; rvecs; tvecs];
                    % Note, the internal reprojection function uses a
                    % different definition of skew factor, 
                    % i.e., s = S / fc(1)
                    [Xp, dXpdr, dXpdt, dXpdf, dXpdc, dXpdkr, dXpdkt, dXpds] = ...
                        visionReprojectPointToSingleCamera(worldPointsXYZ, ...
                                paramStruct.rotationVectors(i, :), ...
                                paramStruct.translationVectors(i, :), ...
                                paramStruct.focalLength, ...
                                paramStruct.principalPoint, ...
                                paramStruct.radialDistortion, ...
                                paramStruct.tangentialDistortion, ...
                                paramStruct.skew/paramStruct.focalLength(1));
                    % Xp: reprojections, 2xN matrix
                    %
                    % dXpdr: Derivative of xp w.r.t rotation vector, 3x(2N)
                    % matrix
                    %
                    % dXpdt: Derivative of xp w.r.t translation vector,
                    % 3x(2N) matrix
                    %
                    % dXpdf: Derivative of xp w.r.t focal length, 2x(2N)
                    % matrix
                    %
                    % dXpdc: Derivative of xp w.r.t principal points,
                    % 2x(2N) matrix
                    %
                    % dXpdkr: Derivative of xp w.r.t radial distortion,
                    % 3x(2N) matrix if there are three coefficients
                    %
                    % dXpdkt: Derivative of xp w.r.t tangential distortion,
                    % 2x(2N) matrix
                    %
                    % dXpds: Derivative of xp w.r.t skew, (2N)x1 vector
                    reprojectedPoints(:,:,i) = Xp';
                    
                    ind = ((i-1)*2*numWorldPoints+1) : (i*2*numWorldPoints);
                    
                    jacobian(ind, 1:2) = dXpdf';
                    jacobian(ind, 3:4) = dXpdc';
                    
                    col = 5;
                    if this.EstimateSkew
                        jacobian(ind, col) = dXpds' / paramStruct.focalLength(1);
                        col = col + 1;
                    end
                    
                    jacobian(ind, col:col+numRadialDistortionCoeffs-1) = dXpdkr';
                    col = col + numRadialDistortionCoeffs;
                    
                    if this.EstimateTangentialDistortion
                        jacobian(ind, col:col+1) = dXpdkt';
                        col = col + 2;
                    end
                    
                    jacobian(ind, [col+i-1,col+i+numPatterns-1,col+i+2*numPatterns-1]) = dXpdr';
                    
                    jacobian(ind, [col+i+numPatterns*3-1,col+i+numPatterns*4-1,col+i+numPatterns*5-1]) = dXpdt';                    
                end
                reprojectedPoints = arrangeImagePointsIntoMatrix(reprojectedPoints);
                jacobian = [jacobian(1:2:end,:);jacobian(2:2:end,:)];
            end           
            
            %----------------------------------------------------------------------
            function pointMatrix = arrangeImagePointsIntoMatrix(imagePoints)
                pointMatrix = reshape(permute(imagePoints, [2, 1, 3]), ...
                    [2, size(imagePoints, 1) * size(imagePoints, 3)])';
            end
        end
        
        %------------------------------------------------------------------
        % Convert the parameter object into a flat parameter vector
        % to be used in optimization.
        %------------------------------------------------------------------
        function x = serialize(this)
            % x = [fx; fy; cx; cy; skew; radial; tangential; rvecs; tvecs];
            
            x = [this.IntrinsicMatrixInternal(1,1); this.IntrinsicMatrixInternal(2,2); ...
                this.IntrinsicMatrixInternal(1,3); this.IntrinsicMatrixInternal(2,3)];
            
            if this.EstimateSkew
                x = [x; this.IntrinsicMatrixInternal(1,2)];
            end
            
            x = [x; this.RadialDistortion(1:this.NumRadialDistortionCoefficients)'];
            
            if this.EstimateTangentialDistortion
                x = [x; this.TangentialDistortion'];
            end
            
            x = [x; this.RotationVectors(:)];
            x = [x; this.TranslationVectors(:)];
        end
        
        
        %------------------------------------------------------------------
        % Initialize the parameter object from a flat parameter vector
        %------------------------------------------------------------------
        function deserialize(this, x)
            paramStruct = unpackSerializedParams(this, x);
            this.IntrinsicMatrixInternal = ...
                vision.internal.calibration.constructIntrinsicMatrix(...
                paramStruct.focalLength(1), paramStruct.focalLength(2), ...
                paramStruct.principalPoint(1), paramStruct.principalPoint(2), ...
                paramStruct.skew);
            
            this.RadialDistortion = paramStruct.radialDistortion;
            this.TangentialDistortion = paramStruct.tangentialDistortion;
            
            this.RotationVectors = paramStruct.rotationVectors;
            this.TranslationVectors = paramStruct.translationVectors;
        end
        
    end
    
    methods(Hidden)
        function paramStruct = unpackSerializedParams(this, x)
            if this.EstimateSkew
                paramStruct.skew = x(5);
                numIntrinsicMatrixEntries = 5;
            else
                paramStruct.skew = 0;
                numIntrinsicMatrixEntries = 4;
            end
            
            paramStruct.focalLength(1) = x(1);
            paramStruct.focalLength(2) = x(2);
            paramStruct.principalPoint(1) = x(3);
            paramStruct.principalPoint(2) = x(4);

            x = x(numIntrinsicMatrixEntries+1:end);
            
            numRadialCoeffs = this.NumRadialDistortionCoefficients;
            paramStruct.radialDistortion = x(1:numRadialCoeffs)';
            
            if this.EstimateTangentialDistortion
                paramStruct.tangentialDistortion = ...
                    x(numRadialCoeffs+1:numRadialCoeffs+2)';
                numDistortionCoeffs = numRadialCoeffs + 2;
            else
                paramStruct.tangentialDistortion = [0,0];
                numDistortionCoeffs = numRadialCoeffs;
            end
            
            x = x(numDistortionCoeffs+1:end);
            
            if isempty(x)
                paramStruct.rotationVectors = [];
                paramStruct.translationVectors = [];
            else
                sizeVecs = length(x) / 2;
                numImages = sizeVecs / 3;
                rvecs = x(1:sizeVecs);
                tvecs = x(sizeVecs+1:end);
                paramStruct.rotationVectors = reshape(rvecs, [numImages, 3]);
                paramStruct.translationVectors = reshape(tvecs, [numImages, 3]);
            end
        end
    end
    
    %----------------------------------------------------------------------
    % saveobj and loadobj are implemented to ensure compatibility across
    % releases even if architecture of the class changes
    methods (Hidden)
       
        function that = saveobj(this)
            that.RadialDistortion = this.RadialDistortion;     
            that.TangentialDistortion = this.TangentialDistortion;
            that.WorldPoints = this.WorldPoints;
            that.WorldUnits = this.WorldUnits;  
            that.EstimateSkew = this.EstimateSkew;
            that.NumRadialDistortionCoefficients = this.NumRadialDistortionCoefficients;
            that.EstimateTangentialDistortion = this.EstimateTangentialDistortion;
            that.RotationVectors = this.RotationVectors;
            that.TranslationVectors = this.TranslationVectors;
            that.ReprojectionErrors = this.ReprojectionErrors;            
            that.IntrinsicMatrix = this.IntrinsicMatrix;
            that.Version = this.Version;
        end
        
    end
    
    
    %--------------------------------------------------------------------------
    
    methods (Static, Hidden)
        
        function this = loadobj(that)
            if isempty(that.ReprojectionErrors)
                reprojErrors = zeros(0, 2, 0);
            else
                reprojErrors = that.ReprojectionErrors;
            end
            
            this = cameraParameters(...
                'IntrinsicMatrix', that.IntrinsicMatrix,...
                'RadialDistortion', that.RadialDistortion,...
                'TangentialDistortion', that.TangentialDistortion,...
                'WorldPoints', that.WorldPoints,...
                'WorldUnits',  that.WorldUnits,...
                'EstimateSkew', that.EstimateSkew,...
                'NumRadialDistortionCoefficients', that.NumRadialDistortionCoefficients,...
                'EstimateTangentialDistortion', that.EstimateTangentialDistortion,...
                'RotationVectors', that.RotationVectors,...
                'TranslationVectors', that.TranslationVectors, ...
                'ReprojectionErrors', reprojErrors);
        end
        
    end
    

end