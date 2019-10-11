function dmat = compute_landmark_transform_min_abs(dmat)

dmat.data = min(abs(dmat.data),[],1);
