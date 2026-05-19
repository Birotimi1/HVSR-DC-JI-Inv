% we actually do not need this anymore 
%c6 is the old Gibbs update for sigma — this is the one that caused the sigma blowup problem.
%We replaced Gibbs with Metropolis-Hastings proposals for sigma, which is what c7 currently does inline (log-normal proposal, MH accept/reject).

function sigma_e_new = c6_gibbs_update_sigma(Phi, N_d, sigma_e_current, prior)

n_noise = prior.noise.n_noise;

sigma_e_new = sigma_e_current(:);

if numel(sigma_e_new) ~= n_noise
    error('c6_gibbs_update_sigma: sigma_e_current length (%d) must equal n_noise (%d).', ...
        numel(sigma_e_new), n_noise);
end
if numel(Phi) ~= n_noise || numel(N_d) ~= n_noise
    error('c6_gibbs_update_sigma: Phi and N_d must have length n_noise (%d).', n_noise);
end

alpha0 = prior.noise.alpha_0(:);
beta0  = prior.noise.beta_0(:);

if numel(alpha0) ~= n_noise || numel(beta0) ~= n_noise
    error('c6_gibbs_update_sigma: prior.noise.alpha_0 and beta_0 must have length n_noise (%d).', n_noise);
end

for d = 1:n_noise
    if ~(N_d(d) > 0 && isfinite(Phi(d)) && Phi(d) >= 0)
        if ~isfinite(sigma_e_new(d)) || sigma_e_new(d) <= 0
            sigma_e_new(d) = sqrt(max(beta0(d), 1e-10) / max(alpha0(d), 1e-6));
        end
        continue;
    end
    
    alpha_post = alpha0(d) + N_d(d) / 2;
    beta_post  = beta0(d)  + Phi(d) / 2;
    
    alpha_post = max(alpha_post, 1e-6);
    beta_post  = max(beta_post,  1e-10);
    
    y = gamrnd(alpha_post, 1 / beta_post);
    
    if ~(isfinite(y) && y > 0)
        y = gamrnd(alpha_post, 1 / beta_post);
    end
    
    if ~(isfinite(y) && y > 0)
        if alpha_post > 1
            sigma2 = beta_post / (alpha_post - 1);
        else
            sigma2 = beta_post / alpha_post;
        end
    else
        sigma2 = 1 / y;
    end
    
    sigma_e_new(d) = sqrt(sigma2);
    
    if ~isfinite(sigma_e_new(d)) || sigma_e_new(d) <= 0
        sigma_e_new(d) = sqrt(max(beta0(d), 1e-10) / max(alpha0(d), 1e-6));
    end
end

end
