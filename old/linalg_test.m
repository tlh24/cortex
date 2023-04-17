g2 = randn(1e4, 4); 
g2 = min(g2, 1.0); 
g2 = max(g2, 0.0); 

w_gen = randn(8, 4) * (1.0 / sqrt(3.0)); 
gen = g2*w_gen'; 
gen = min(gen, 1.5); 
gen = max(gen, 0.0); 
invgen = 1.0 - gen; 
catgen = [gen invgen]; 

% given the input-output mapping, how much of the input can we predict from
% the output? (function inversion)
m = g2\catgen; 
pred = g2*m; 
err = catgen - pred; 
mean(err, 1)

% clearly easy to predict the linear 'gen' variables, 
% but not easy to predict the invgen variables. 