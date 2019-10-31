K = 50;
D = 2;
T = 100;
dampen = .25;
sigmareal = .1;
sigmaobs = .0;
appearancemu = .9;
appearancesig = .2;
badappearancemu = .1;
badappearancesig = .2;
poslambda = .1;
misscost = (appearancemu+badappearancemu)/2 + sigmareal*poslambda;

pmiss = .3;

Xreal = nan(D,T);
noise = randn([D,T])*sigmareal;
Xreal(:,1) = noise(:,1);
Xreal(:,2) = Xreal(:,1) + noise(:,2);
for t = 3:T,
  v = Xreal(:,t-1)-Xreal(:,t-2);
  pred = Xreal(:,t-1) + dampen*v;
  Xreal(:,t) = pred + noise(:,t);
end

minx = min(Xreal,[],2);
maxx = max(Xreal,[],2);

figure(1);
clf;
PlotInterpColorLine(Xreal(1,:),Xreal(2,:),jet(T),[],'LineWidth',2);
set(gca,'Color','k');

Xobs = rand(D,T,K).*(maxx-minx) + minx;
appearancescore = max(0,min(1,randn(T,K)*badappearancesig+badappearancemu));
ismiss = rand(1,T) <= pmiss;
ismiss(1:2) = false; % TO DO: remove this
samplereal = randsample(K,T,true);
noise = randn([D,T])*sigmaobs;
for t = 1:T,
  if ~ismiss(t),
    Xobs(:,t,samplereal(t)) = Xreal(:,t) + noise(:,t);
    appearancescore(t,samplereal(t)) = randn(1)*appearancesig + appearancemu;
  end  
end

hold on;

color = reshape(jet(T),[T,1,3]) .* appearancescore;
scatter(vectorize(Xobs(1,:,:)),vectorize(Xobs(2,:,:)),[],reshape(color,[T*K,3]),'o','filled');
colormap(jet(T)*.75);


plot(Xreal(1,~ismiss),Xreal(2,~ismiss),'wo');
plot(Xreal(1,ismiss),Xreal(2,ismiss),'wx','LineWidth',2);
[t0s,t1s] = get_interval_ends(ismiss);
nmiss = t1s-t0s

appearancecost = 1-appearancescore;

global realinfo;
realinfo.Xreal = Xreal;
realinfo.samplereal = samplereal;
realinfo.ismiss = ismiss;

%% inference

[Xbest,vbest,idx,totalcost,poslambda] = ...
  ChooseBestTrajectory_MissDetection(Xobs,appearancecost,...
  'dampen',dampen,...
  'poslambda',poslambda,...
  'misscost',misscost);

samplerealmiss = samplereal;
samplerealmiss(ismiss) = K+1;

XrealMiss = Xreal;
XrealMiss(:,ismiss) = nan;

%% plot results

figure(4);
clf;
for d = 1:2,
  subplot(2,2,(d-1)*2+1);
  plot(1:T,Xreal(d,:),'b--','LineWidth',2);
  hold on;
  plot(1:T,XrealMiss(d,:),'b-','LineWidth',2);
  plot(1:T,Xbest(d,:),'-','Color',[.7,0,0],'LineWidth',2);
  plot(find(~ismiss),Xreal(d,~ismiss),'bo','LineWidth',2);
  plot(find(ismiss),Xreal(d,ismiss),'bx','LineWidth',2);
  plot(find(vbest==1),Xbest(d,vbest==1),'rs','LineWidth',2);
  plot(find(vbest==0),Xbest(d,vbest==0),'r+','LineWidth',2);
  scatter(vectorize(repmat(1:T,[1,1,K])),vectorize(Xobs(d,:,:)),max(eps,min(1,(1-appearancecost(:))))*30,'k','o','filled');
  if d == 1,
    ylabel('x');
  else
    ylabel('y');
  end
  axisalmosttight;
  subplot(2,2,(d-1)*2+2);
  scatter(vectorize(repmat(1:T,[1,1,K])),vectorize(Xobs(d,:,:)),max(eps,min(1,(1-appearancecost(:))))*30,'k','o','filled');
   if d == 1,
    ylabel('x');
  else
    ylabel('y');
  end
  axisalmosttight;
end

%% now let's try on some real data?

frontviewmatfile = '/groups/huston/hustonlab/flp-chrimson_experiments/tempTrackingOutput/hustonlab__fly492__0019_c_front.mat';
sideviewmatfile = '/groups/huston/hustonlab/flp-chrimson_experiments/tempTrackingOutput/hustonlab__fly492__0019_c_side.mat';
kinematfile = '/groups/huston/hustonlab/flp-chrimson_experiments/fly_460_to_462_5_3_17_norpASS00325/calib_5_3_2017/checkerBoard_200micronSquares_5_3_17/orthocam_strocal_5_3_17.mat';
