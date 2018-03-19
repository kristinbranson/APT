function [roixlo,roixhi,roiylo,roiyhi] = xyRad2roi(x,y,roiradius)
xrnd = round(x);
yrnd = round(y);
roixlo = xrnd - roiradius;
roixhi = xrnd + roiradius;
roiylo = yrnd - roiradius;
roiyhi = yrnd + roiradius;