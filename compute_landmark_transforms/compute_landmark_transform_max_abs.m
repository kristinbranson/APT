function dmat = compute_landmark_transform_max_abs(dmat)

dmat.data = max(abs(dmat.data),[],1);
