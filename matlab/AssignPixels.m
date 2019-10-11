function [cc,nflies] = AssignPixels(isfore,diffim,trx,t)

[y,x] = find(isfore);
X = [x(:),y(:)];
w = diffim(isfore);
mu = zeros(0,2);
S = zeros(2,2,0);
for fly = 1:numel(trx)
  if t > trx(fly).endframe || t < trx(fly).firstframe
    continue;
  end
  i = trx(fly).off + t;
  mu(end+1,:) = [trx(fly).x(i),trx(fly).y(i)];  %#ok<AGROW>
  S(:,:,end+1) = axes2cov(trx(fly).a(i)*2,trx(fly).b(i)*2,trx(fly).theta(i)); %#ok<AGROW>
end
nflies = size(mu,1);

if nflies == 0
  cc = zeros(size(isfore));
  return;
end
mix = gmm(2,nflies,'full');
mix.centres = mu;
mix.covars = S;

[~,~,~,post] = mygmm(X,nflies,'start',mix,'weights',w);
[~,l] = max(post,[],2);
cc = zeros(size(isfore));
cc(isfore) = l;