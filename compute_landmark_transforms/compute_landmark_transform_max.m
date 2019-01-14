function dmat = compute_landmark_transform_max(dmat)

dmat.data = max(dmat.data,[],1);
