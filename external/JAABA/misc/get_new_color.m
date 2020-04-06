function color = get_new_color(i)

N = 8;
TWON = 2^N;

global GETNEWCOLOR___;
if isempty(GETNEWCOLOR___),
  GETNEWCOLOR___ = jet(TWON);
end
i = mod(i-1,TWON)+1;

if i == 1,
  color = GETNEWCOLOR___(1,:);
  return;
end
if i == 2,
  color = GETNEWCOLOR___(end,:);
  return;
end

total = 2;
for j = 1:8,
end

v = floor(linspace(1,256,total));
offset = i - (total - j);
l = v(2*offset);

color = GETNEWCOLOR___(l,:);
