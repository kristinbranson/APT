function op_affinity_graph_str = skel2aptcmdilneconfopt(skel)
% op_affinity_graph_str = skel2aptcmdilneconfopt(skel)
%
% skel: [nEdge x 2] skeletonEdges, 1-based
%
% op_affinity_graph_str: cmdline conf_param arg for APT_interf

[nedge,d] = size(skel);
assert(d==2);

skel = sortrows(skel);
skel = skel-1;

s = '';
for iedge=1:nedge
  stmp = sprintf('\\(%d,%d\\),',skel(iedge,1),skel(iedge,2));
  s = [s stmp]; %#ok<AGROW>
end

s = s(1:end-1); % rm trailing comma
op_affinity_graph_str = s;
