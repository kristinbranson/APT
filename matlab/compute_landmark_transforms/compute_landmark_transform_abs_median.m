function dmat = compute_landmark_transform_abs_median(dmat)

dmat.data = abs(median(dmat.data,1));
