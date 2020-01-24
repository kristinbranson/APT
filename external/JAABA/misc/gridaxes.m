% hax = gridaxes(r0,r1,c0,c1)
function hax = gridaxes(r0,r1,c0,c1)

clf;
nr = length(r0);
nc = length(c0);
hax = zeros(nr,nc);

for r = 1:nr,
  top = 1 - r0(r);
  height = r1(r) - r0(r);
  bottom = top - height;
  for c = 1:nc,
    left = c0(c);
    width = c1(c) - c0(c);
    hax(r,c) = axes('position',[left,bottom,width,height]);
  end
end