function triangulated = multiDLT(x_s, proj_mats, fc_s, cc_s, kc_s, alpha_cs)
    % multi camera DLT
    
    %--- Normalize the image projection according to the intrinsic parameters of the left and right cameras
    x_norms = cell(size(x_s, 3), 1);
    for i = 1:size(x_s, 3)
        x_norms{i} = normalize_pixel(x_s(:, :, i), fc_s{i}, cc_s{i}, kc_s{i}, alpha_cs{i});
    end
    
    %--- Number of points:
    num_points = size(x_norms{1}, 2);
    num_views = length(x_norms);
    
    triangulated = zeros(3, num_points); 
    for i = 1:num_points
        % create the system to solve
        A = zeros(num_views * 2, 4);
        for j = 1:num_views
            x = x_norms{j}(:, i);
            A((j - 1) * 2 + 1, :) = x(1) * proj_mats{j}(3, :) - proj_mats{j}(1, :);
            A(j * 2, :) = x(2) * proj_mats{j}(3, :) - proj_mats{j}(2, :);
        end
        % DEBUG
        %A(1:2, :) = nan;
        % remove rows with nans
        nan_idx = false(num_views * 2, 1);
        for j = 1:4
            nan_idx = nan_idx | isnan(A(:, j));
        end
        A = A(~nan_idx, :);
    
        % if there are no valid rows, put in nans.
        if isempty(A)
            triangulated(:, i) = nan(3, 1);
        else
            % if A only has 2 rows (ie only one labeled point) does this still make sense to do?
            [~, ~, v] = svd(A);
            temp = v(:, end) / v(end, end);
            triangulated(:, i) = temp(1:3);
        end
    end