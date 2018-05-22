K = 50;
D = 2;
T = 100;
dampen = .25;
sigmareal = .1;
sigmaobs = 0;
appearancemu = .9;
appearancesig = .01;
badappearancemu = .1;
badappearancesig = .01;
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

%%
[Xbest,vbest,idx,totalcost,poslambda] = ...
  ChooseBestTrajectory_MissDetection(Xobs,appearancecost,...
  'dampen',dampen,...
  'poslambda',poslambda,...
  'misscost',misscost);

samplerealmiss = samplereal;
samplerealmiss(ismiss) = K+1;

%%

figure(3);
clf;
PlotInterpColorLine(Xreal(1,:),Xreal(2,:),jet(T),[],'LineWidth',2,'LineStyle','--');
hold on;
XrealMiss = Xreal;
XrealMiss(:,ismiss) = nan;
PlotInterpColorLine(XrealMiss(1,:),XrealMiss(2,:),jet(T),[],'LineWidth',2);
plot(Xreal(1,~ismiss),Xreal(2,~ismiss),'mo','LineWidth',2);
plot(Xreal(1,ismiss),Xreal(2,ismiss),'cx','LineWidth',2)
plot(Xbest(1,:),Xbest(2,:),'k:','LineWidth',2)
plot(Xbest(1,vbest==1),Xbest(2,vbest==1),'ks','LineWidth',2)
plot(Xbest(1,vbest==0),Xbest(2,vbest==0),'k+','LineWidth',2)


