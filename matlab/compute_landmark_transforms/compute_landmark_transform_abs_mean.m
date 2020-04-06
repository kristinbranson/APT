function dmat = compute_landmark_transform_abs_mean(dmat)

dmat.data = abs(mean(dmat.data,1));
