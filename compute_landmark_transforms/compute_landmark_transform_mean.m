function dmat = compute_landmark_transform_mean(dmat)

dmat.data = mean(dmat.data,1);
