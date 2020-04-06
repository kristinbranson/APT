% [edges,centers] = SelectHistEdges(nbins,lim,binmode)
% It selects edges between bounds set by LIM according to BINMODE, 
% which is either 'linear', 'log' (log binning), or 'logabs' 
% (log binning in both directions from zero).

function [edges,centers] = SelectHistEdges(nbins,lim,binmode)

switch lower(binmode),
  case 'linear',
    edges = linspace(lim(1),lim(2),nbins+1);
  case 'log',
    % make sure > 0

    off = (lim(2)-lim(1))/nbins;
    tmplim = lim - lim(1) + off;
    edges = exp(linspace(log(tmplim(1)),log(tmplim(2)),nbins+1));
    edges = edges + lim(1) - off;
      
  case 'logabs',
    % from lim(1) to 0, we do log spacing on neg value
    % from 0 to lim(2), we do log spacing on pos value
    
    if lim(1) > 0,
      % corner case: both are positive
      tmplim = lim;
      edges = exp(linspace(log(tmplim(1)),log(tmplim(2)),nbins+1));
      
    elseif lim(2) < 0,
      % corner case: both are negative
      tmplim = -lim;
      edges = exp(linspace(log(tmplim(1)),log(tmplim(2)),nbins+1));
      edges = fliplr(-edges);
      
    else
      % how much of data is below 0
      fracneg = -lim(1) / (lim(2)-lim(1));
      fracpos = 1 - fracneg;
      % how many bins will we have on one side of 0
      if fracneg > .5,
        nbinsneg = floor(fracneg*nbins);
        nbinspos = nbins - nbinsneg;
      else
        nbinspos = floor(fracpos*nbins);
        nbinsneg = nbins - nbinspos;
      end
      % positive edges
      off = (lim(2)-lim(1))/nbins;
      tmplim = [0,lim(2)] + off;
      edgespos = exp(linspace(log(tmplim(1)),log(tmplim(2)),nbinspos+1));
      edgespos = edgespos - off;
      % negative edges
      tmplim = [0,-lim(1)];
      tmplim = tmplim + off;
      edgesneg = exp(linspace(log(tmplim(1)),log(tmplim(2)),nbinsneg+1));
      edgesneg = fliplr(-(edgesneg - off));
      edges = [edgesneg(1:end-1),edgespos];
    end
    
  otherwise,
    error('Unknown binmode %s',binmode);
end

centers = (edges(1:end-1)+edges(2:end))/2;