function dmat = compute_landmark_transform_min(dmat)

dmat.data = min(dmat.data,[],1);
