function dmat = compute_landmark_transform_std(dmat)

dmat.data = std(dmat.data,1,1);
