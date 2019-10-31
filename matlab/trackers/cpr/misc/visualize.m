model = shapeGt('createModel','larva');

% VISUALIZE Example results on a test image
figure,clf,
nimage=10;
%Ground-truth
subplot(1,2,1),
shapeGt('draw',model,IsT{nimage},phisT(nimage,:),{'lw',20});
title('Ground Truth');
%Prediction
subplot(1,2,2),shapeGt('draw',model,IsT{nimage},pT(nimage,:),...
    {'lw',20});
title('Prediction');
%% VISUALIZE Example results on a TRAINING image
figure,clf,
nimage=10;
%Ground-truth
subplot(1,2,1),
shapeGt('draw',model,IsTr{nimage},phisTr(nimage,:),{'lw',20});
title('Ground Truth');
%Prediction
subplot(1,2,2),shapeGt('draw',model,IsTr{nimage},pTr(nimage,:),...
    {'lw',20});
title('Prediction');