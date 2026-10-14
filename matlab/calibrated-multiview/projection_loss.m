function loss = projection_loss(X, x_org, Rs, Ts, fc_s, cc_s, kc_s, alpha_cs)

    % assumes X is in the first frame's view
    num_views = length(fc_s);

    loss = 0;
    % make it a weighted loss?
    for i = 1:num_views
        if ~any(isnan(x_org(:, :, i)))
            reprojections = project_points2(X, rodrigues(Rs{i}), Ts{i}, fc_s{i}, cc_s{i}, kc_s{i}, alpha_cs{i});
            loss = loss + sum(sqrt(sum((x_org(:, :, i) - reprojections) .* (x_org(:, :, i) - reprojections), 1)));
        end
    end
end