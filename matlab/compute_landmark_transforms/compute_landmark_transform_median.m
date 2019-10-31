function dmat = compute_landmark_transform_median(dmat)

dmat.data = median(dmat.data,1);
