function y = meanignorenan(x)

y = mean(x(~isnan(x)));