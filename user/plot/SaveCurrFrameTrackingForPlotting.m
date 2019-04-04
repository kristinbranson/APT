function SaveCurrFrameTrackingForPlotting(lObj,savefilein)

persistent savefile;
if nargin >= 2,
  savefile = savefilein;
end
if isempty(savefile),
  savefile = '';
end
  
t = lObj.currFrame;
iTarg = lObj.currTarget;
iMov = lObj.currMovie;

nViews = lObj.nview;
currIm = cell(1,nViews);
currImRoi = zeros(4,nViews);

for iView = 1:nViews,
  
  currIm{iView} = get(lObj.gdata.images_all(iView),'CData');
  currImRoi(1:2,iView) = get(lObj.gdata.images_all(iView),'XData');
  currImRoi(3:4,iView) = get(lObj.gdata.images_all(iView),'YData');
  
end

tObj = lObj.tracker;

% Use loaded predictions by default
xyPredictions = lObj.labeledpos2{iMov};
if all(isnan(xyPredictions))
  xyPredictions = tObj.trkP; % [npt x d x ntgt]
end

idxPred = find(~isnan(xyPredictions));
predIdx = struct;
[predIdx.pt,predIdx.d,predIdx.frm,predIdx.tgt] = ind2sub(size(xyPredictions),idxPred);
xyPredictions = xyPredictions(idxPred);


axPropNames = {
  'CameraPosition'
  'CameraPositionMode'
  'CameraTarget'
  'CameraTargetMode'
  'CameraUpVector'
  'CameraUpVectorMode'
  'CameraViewAngle'
  'CameraViewAngleMode'
  'View'
  'XLim'
  'XLimMode'
  'YLim'
  'YLimMode'
  'ZLim'
  'ZLimMode'
  'XDir'
  'YDir'
  'ZDir'
  'CLim'
  };
imPropNames = {'XData','YData'};
imProps = cell(1,nViews);
axProps = cell(1,nViews);
for i = 1:nViews,
  axProps{i} = get(lObj.gdata.axes_all(i),axPropNames);
  imProps{i} = get(lObj.gdata.images_all(i),imPropNames);
end

if nargin < 2,
  [f,p] = uiputfile(savefile,'Save data to mat file');
  if ~ischar(f),
    return;
  end
  savefile = fullfile(p,f);
end

colors = lObj.labelPointsPlotInfo.Colors;

save(savefile,'currIm','currImRoi','xyPredictions','predIdx','t','iTarg','iMov','axProps','axPropNames','colors','imProps','imPropNames');
