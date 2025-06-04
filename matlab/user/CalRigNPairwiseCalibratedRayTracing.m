classdef CalRigNPairwiseCalibratedRayTracing < CalRig & matlab.mixin.Copyable
    % N-camera rig, with pairwise Caltech calibs

    %CalRig
    properties
        nviews;
        viewNames;
    end
    properties (Dependent)
        ncams % same as nviews
    end
    properties (Constant)
        conda_env = 'APT_raytracing';
    end

    methods
        function v = get.ncams(obj)
            v = obj.nviews;
        end
    end

    properties
        % [ncams,ncams] cell array of pairwise calrig objs
        % This is a (strict) upper-triangular mat with an empty diagonal.
        crigStros
        % model_path = '/groups/branson/bransonlab/aniket/fly_walk_imaging/calibration_code/refraction_model/calprism/pyCall/model_two_cameras_prism.pth';
        % python_script_path = '/groups/branson/bransonlab/aniket/fly_walk_imaging/calibration_code/refraction_model/calprism/pyCall/return_projected_ray_two_cameras_prism.py';
        model_path
        python_script_path
        dividing_col
        image_width
    end

    methods

        function obj = CalRigNPairwiseCalibratedRayTracing(varargin)
            if nargin==1
                if isstruct(varargin{1})
                    s = varargin{1};
                end
            end

            [tfSuccess,msg] = CalRigNPairwiseCalibratedRayTracing.setPythonEnv();
            if ~tfSuccess,
              warningNoTrace(msg);
            end
            obj.nviews = s.nviews;
            if isfield(s,'crigStros'),
              obj.crigStros = s.crigStros;
            else
              obj.crigStros = s.calibrations;
            end
            % The following two lines were added by Aniket Ravan
            obj.model_path = s.model_path;
            % obj.python_script_path = s.python_script_path;
            obj.python_script_path = fullfile(APT.Root, 'raytracing_calib', 'programs', 'return_projected_ray_two_cameras_prism.py');
            obj.dividing_col = s.dividing_col;
            obj.image_width = s.image_width;
            % c = 1;
            % % ordering of stereo crigs assumed
            % for icam=1:ncam
            % for jcam=icam+1:ncam
            %   obj.crigStros{icam,jcam} = crigs{c};
            %   c = c+1;lObj
            % end
            % end
            %assert(c==numel(crigs)+1);

            obj.viewNames = arrayfun(@(x)sprintf('view%d',x),(1:obj.nviews)','uni',0);
        end
    end

    methods

        % Helper. Get stereo crig and map "absolute" cam indices to left/right
        function [crigstro,ivwLstro,ivwRstro] = getStroCalRig(obj,iView1,iView2)
            % iView1/2: view indices
            %
            % crigstro: calrig object for iview1,2 pair
            % ivwLstro: 1 if iView1 maps to "left", 2 if "right"
            % ivwRstro: "    iView2  "

            if iView1<iView2
                crigstro = obj.crigStros{iView1,iView2};
                ivwLstro = 1;
                ivwRstro = 2;
            elseif iView2<iView1
                crigstro = obj.crigStros{iView2,iView1};
                ivwLstro = 2;
                ivwRstro = 1;
            else
                assert(false);
            end
        end

        function cam_label = viewIdToLabel(obj, iView1)
            % Aniket Ravan, 4th of Feb 2025
            % Converts a pair of View indices (used in APT) to camera labels (used in Pytorch ray
            % tracing-based calibration)
            % Convention: 1-primary camera (virtual)
            %             2-primary camera (real)
            %             3-secondary camera (virtual)
            %             4-secondary camera (real)

            switch iView1
                case 1
                    cam_label = "primary_virtual";
                case 2
                    cam_label = "primary_real";
                case 3
                    cam_label = "secondary_virtual";
                case 4
                    cam_label = "secondary_real";
            end

        end

        function [xEPL, yEPL] = rayTracingComputeEpipolarLine_old(obj, iView1, iView2, xy1)
            % Aniket Ravan, 4th of Feb 2025
            % View id convention: 1-primary camera (virtual)
            %                     2-primary camera (real)
            %                     3-secondary camera (virtual)
            %                     4-secondary camera (real)
            % iView1: View that has been labelled
            % iView2: View on which epipolar line is drawn

            if mod(iView2, 2) ~= 0 % If epipolar line is drawn on virtual camera
                if mod(iView1, 2) ~= 0 % If virtualcamera is labelled
                    % fprintf('Note: Epipolar lines are only available on real views as of now\n')
                    xEPL = xy1(1);
                    yEPL = xy1(1);
                else
                    switch iView2
                        case 1
                            if iView1 == 2
                                cam_label = obj.viewIdToLabel(iView1);
                                [~, epipolar_line_labelled] = getEPLRayTracing(obj.model_path,...
                                    obj.python_script_path, ...
                                    obj.dividing_col,...
                                    obj.image_width,...
                                    cam_label,...
                                    xy1);
                                xEPL = epipolar_line_labelled(1,:);
                                yEPL = epipolar_line_labelled(2,:);
                            elseif iView1 == 4
                                xEPL = 1;
                                yEPL = 1;
                            end
                        case 3
                            if iView1 == 2
                                xEPL = 1;
                                yEPL = 1;
                            elseif iView1 == 4
                                cam_label = obj.viewIdToLabel(iView1);
                                [~, epipolar_line_labelled] = getEPLRayTracing(obj.model_path,...
                                    obj.python_script_path, ...
                                    obj.dividing_col,...
                                    obj.image_width,...
                                    cam_label,...
                                    xy1);
                                xEPL = epipolar_line_labelled(1,:);
                                yEPL = epipolar_line_labelled(2,:);
                            end
                    end

                end
            
            else % if epipolar line is being drawn in real view

                cam_label = obj.viewIdToLabel(iView1);
                [epipolar_line_unlabelled, epipolar_line_labelled] = getEPLRayTracing(obj.model_path,...
                    obj.python_script_path, ...
                    obj.dividing_col,...
                    obj.image_width,...
                    cam_label,...
                    xy1);

                % format long epipolar_line_labelled
                switch iView2
                    % Cases 1 and 3 are not implemented yet. Error message written above
                    case 2

                        if iView1 == 1
                            xEPL = epipolar_line_labelled(1,:);
                            yEPL = epipolar_line_labelled(2,:);

                        elseif iView1 == 3 || iView1 == 4
                            xEPL = epipolar_line_unlabelled(1,:);
                            yEPL = epipolar_line_unlabelled(2,:);
                        end
                    case 4
                        if iView1 == 1 || iView1 == 2
                            xEPL = epipolar_line_unlabelled(1,:);
                            yEPL = epipolar_line_unlabelled(2,:);
                        elseif iView1 == 3
                            xEPL = epipolar_line_labelled(1,:);
                            yEPL = epipolar_line_labelled(2,:);
                        end
                end
            end
        end

        function [xEPL, yEPL] = rayTracingComputeEpipolarLine(obj, iView1, iView2, xy1)
            % Aniket Ravan, 4th of Feb 2025
            % View id convention: 1-primary camera (virtual)
            %                     2-primary camera (real)
            %                     3-secondary camera (virtual)
            %                     4-secondary camera (real)
            % iView1: View that has been labelled
            % iView2: View on which epipolar line is drawn
            cam_label = obj.viewIdToLabel(iView1);
            cam_projecting = obj.viewIdToLabel(iView2);
            epipolar_line = getEPLRayTracing(obj.model_path, ...
                                    obj.python_script_path, ...
                                    obj.dividing_col,...
                                    obj.image_width,...
                                    cam_label, ...
                                    cam_projecting,...
                                    xy1);
            xEPL = epipolar_line(1,:);
            yEPL = epipolar_line(2,:);
        end

        %CalRig
        function [xEPL,yEPL] = computeEpiPolarLine(obj,iView1,xy1,iViewEpi,imroi)
            % Aniket Ravan, 4th of Feb 2025
            %[crigstro,ivwLstro,ivwRstro] = obj.getStroCalRig(iView1,iViewEpi); % getStroCalRig gets the meta data for the two cameras involved
            %[xEPL,yEPL] = crigstro.computeEpiPolarLine(ivwLstro,xy1,ivwRstro,imroi); % computeEpipolarLine computes the epipolar line given the meta data
            [xEPL, yEPL] = obj.rayTracingComputeEpipolarLine(iView1, iViewEpi, xy1);
        end

        function [X,xprp,rpe] = triangulate(obj,xp,withFmin)
            [~, num_points, num_views] = size(xp);
            assert(obj.ncams == num_views);
            if nargin < 3
                withFmin = false;
            end

            % construct projection matricies between the cameras.
            % use the first camera as source/world frame of reference.
            all_projections = cell(obj.nviews,1);
            all_projections{1} = [eye(3), zeros(3, 1)];

            fc_s = cell(obj.nviews,1);
            cc_s = cell(obj.nviews,1);
            kc_s = cell(obj.nviews,1);
            alpha_cs = cell(obj.nviews,1);

            fc_s{1} = obj.crigStros{1,2}.int.L.fc;
            cc_s{1} = obj.crigStros{1,2}.int.L.cc;
            kc_s{1} = obj.crigStros{1,2}.int.L.kc;
            alpha_cs{1} = obj.crigStros{1,2}.int.L.alpha_c;

            for i=2:obj.nviews
                all_projections{i} = [obj.crigStros{1,i}.R.LR,obj.crigStros{1,i}.TLR];
                fc_s{i} = obj.crigStros{1,i}.int.R.fc;
                cc_s{i} = obj.crigStros{1,i}.int.R.cc;
                kc_s{i} = obj.crigStros{1,i}.int.R.kc;
                alpha_cs{i} = obj.crigStros{1,i}.int.R.alpha_c;
            end

            X = multiDLT(xp, all_projections, fc_s, cc_s, kc_s, alpha_cs);
            % create the transformations between each view.
            Rs = cell(obj.nviews, 1);
            Ts = cell(obj.nviews, 1);
            for i = 1:num_views
                if i == 1
                    Rs{i} = eye(3);
                    Ts{i} = zeros(3, 1);
                else
                    Rs{i} = obj.crigStros{1, i}.R.LR;
                    Ts{i} = obj.crigStros{1, i}.T.LR;
                end
            end

            % does it make sense to do the optimization at a per point level?
            if withFmin
                %options = optimset('MaxFunEvals', 50000);
                X_min = zeros(size(X));
                for i = 1:num_points
                    opt_func = @(X_tri) projection_loss(X_tri, xp(:, i, :), Rs, Ts, fc_s, cc_s, kc_s, alpha_cs);
                    X_min(:, i) = fminsearch(opt_func, X(:, i));
                end
            else
                X_min = X;
            end

            % create the reprojections and rpe
            xprp = zeros(size(xp));
            xprp_dlt = zeros(size(xp));
            for i = 1:num_views
                if i == 1
                    R = eye(3);
                    T = zeros(3, 1);
                    fc = obj.crigStros{1, 2}.int.L.fc;
                    cc = obj.crigStros{1, 2}.int.L.cc;
                    kc = obj.crigStros{1, 2}.int.L.kc;
                    alpha_c = obj.crigStros{1, 2}.int.L.alpha_c;
                else
                    R = obj.crigStros{1, i}.R.LR;
                    T = obj.crigStros{1, i}.T.LR;
                    fc = obj.crigStros{1, i}.int.R.fc;
                    cc = obj.crigStros{1, i}.int.R.cc;
                    kc = obj.crigStros{1, i}.int.R.kc;
                    alpha_c = obj.crigStros{1, i}.int.R.alpha_c;
                end
                xprp(:, :, i) = project_points2(X_min, rodrigues(R), T, fc, cc, kc, alpha_c);
                xprp_dlt(:, :, i) = project_points2(X, rodrigues(R), T, fc, cc, kc, alpha_c);
            end

            rpe = zeros(num_views, 1);
            sumsq = reshape(sum((xp - xprp) .* (xp - xprp), 1), num_points, num_views);
            for i = 1:num_views
                rpe(i) = mean(sqrt(sumsq(~isnan(sumsq(:, i)), i)));
            end
            % %rpe = mean(sum((xp - xprp) .* (xp - xprp), 1));
            % xp_temp = reshape(xp, 2, num_points * num_views);
            % xprp_temp = reshape(xprp, 2, num_points * num_views);

            % % create the non nan indexing
            % nan_idxing = ~(isnan(xp_temp(1, :)) | isnan(xp_temp(2, :)));
            % %keyboard
            % %rpe = mean((xp_temp(:, nan_idxing) - xprp_temp(:, nan_idxing)) .* (xp_temp(:, nan_idxing) - xprp_temp(:, nan_idxing), 1);
            % %rpe = mean(sqrt(sum(, 1)));
            % if any(isnan(rpe))
            %   keyboard
            % end
        end

        %CalRig
        function [xRCT,yRCT] = reconstruct(obj,iView1,xy1,iView2,xy2,iViewRct)

            assert(numel(xy1)==2);
            assert(numel(xy2)==2);

            [crigTri,ivwLTri,ivwRTri] = obj.getStroCalRig(iView1,iView2);
            xy = cat(3,xy1(:),xy2(:));
            idxTri = [ivwLTri ivwRTri]; % [1 2] if iView1/2 are l/r; else [2 1]
            xy = xy(:,:,idxTri);

            iviewsTriangulate = [iView1 iView2];
            iviewLTriangulate = iviewsTriangulate(idxTri(1));
            iviewRTriangulate = iviewsTriangulate(idxTri(2));

            X = crigTri.triangulate(xy); % X in coord sys of crigstro 'L', or iviewLTriangulate
            % Triangulate 2d points into 3d position
            % xy: [2xnxnviews] 2d image points
            % X: [3xn] reconstructed 3d points. coord sys may depend on concrete
            %   subclass. (typically, coord sys of camera 1.)
            ylTri = crigTri.x2y(crigTri.project(X,'L'),'L');
            XrTri = crigTri.camxform(X,'LR');
            yrTri = crigTri.x2y(crigTri.project(XrTri,'R'),'R');
            fprintf(1,'RP pt. View %d: %s. View %d: %s.\n',iviewLTriangulate,...
                mat2str(round(ylTri(end:-1:1))),iviewRTriangulate,mat2str(round(yrTri(end:-1:1))));

            [crigRC,ivwLRC] = obj.getStroCalRig(iviewLTriangulate,iViewRct);
            CAMXFORMS = {'LR' 'RL'};
            camxform = CAMXFORMS{ivwLRC}; % if ivw1RC==1, translate l->r. else translate r->l.
            Xrc = crigRC.camxform(X,camxform); % transform from iviewLTriangulate->iViewRct
            camRCT = camxform(2);
            y = crigRC.x2y(crigRC.project(Xrc,camRCT),camRCT);

            [crigRC2,ivwLRC2] = obj.getStroCalRig(iviewRTriangulate,iViewRct);
            camxform = CAMXFORMS{ivwLRC2};
            Xrc2 = crigRC2.camxform(XrTri,camxform);
            camRCT = camxform(2);
            y2 = crigRC2.x2y(crigRC2.project(Xrc2,camRCT),camRCT);

            xRCT = [y([2 2]) y2(2)];
            yRCT = [y([1 1]) y2(1)];
        end

    end

    methods (Static)

      function [tfSuccess,msg] = setPythonEnv()
        persistent isset
        tfSuccess = true;
        msg = '';
        if ~isempty(isset) && isset,
          return;
        end
        command = sprintf('conda run -n %s which python', CalRigNPairwiseCalibratedRayTracing.conda_env);
        [res, cmdout] = system(command);
        if res ~= 0,
          tfSuccess = false;
          msg = sprintf('Conda environment %s no found: %s', CalRigNPairwiseCalibratedRayTracing.conda_env,strtrim(cmdout));
        end
        python_env_path = strtrim(cmdout); % Remove the end-of-line character
        try
          pyenv('Version', python_env_path,'ExecutionMode','OutOfProcess'); % Initialize python environment
          isset = true;
          % ExecutionMode=OutOfProcess is important to avoid clashes between incompatible MKL libraries between PyTorch and MATLAB
        catch ME
          warningNoTrace('Error setting python environment to %s:\n%s',python_env_path,getReport(ME));
        end
      end
    end    
end