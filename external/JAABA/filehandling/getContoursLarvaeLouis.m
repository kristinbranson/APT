function [contours] = getContoursLarvaeLouis(coeffs,nPoints)
% calculate contours using fourier coefficients
% author: marvin albert
% inputs:
%     kinData.fourier contains coeffs over time (time,ncoeffs)
%     nPoints: contour resolution
% output:
%     contours over time (time,nPoints)

%coeffs = kinData.fourier;
dims = size(coeffs);
contours = zeros(dims(1),nPoints,2);
thetaStep = 2*pi/nPoints;
for timeind=1:dims(1)
    for coeffind=1:(dims(2)/4)
        theta = -pi;
        for pointind=1:nPoints
            for coordind=1:2
                tmpcoeffind = (coeffind-1)*4+1+(coordind-1)*2;
                tmp = coeffs(timeind,tmpcoeffind)*cos((coeffind-1)*theta)+coeffs(timeind,tmpcoeffind+1)*sin((coeffind-1)*theta);
                contours(timeind,pointind,coordind) = contours(timeind,pointind,coordind) + tmp;
            end
            theta = theta + thetaStep;
        end
    end
end