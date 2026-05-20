function log_p = compute_log_prior_sigma(sigma_e, prior)
%
% Log-prior on the hierarchical noise multiplier gamma_d (stored as sigma_e)
%
% SIGMA DEFINITIONS:
%   sigma_e(d) = gamma_d = hierarchical multiplier for data type d
%   gamma_d = 1.0 means trust data errors as given
%   gamma_d < 1.0 squeezes errors (data constrains more)
%   gamma_d > 1.0 expands errors (data constrains less)
%
% PRIOR:
%   Log-normal centered at gamma_d = 1.0 with broad spread
%   The floor (sigma_min) is the critical constraint:
%   prevents any data type from dominating by shrinking its errors too much
%   The ceiling (sigma_max) is generous: gamma_d can grow freely
%   Following Zach's guidance: broad prior, strong floor, loose ceiling

log_p = 0;
sigma_prior = prior.noise.sigma_prior;
sigma_min = prior.noise.sigma_min;
sigma_max = prior.noise.sigma_max;

% broad log-normal width: 2 decades spread
% gamma_d can range from ~0.1 to ~10 before the prior penalizes significantly
log_sigma_width = 2.0;

for d = 1:prior.noise.n_noise
    % skip data types with no valid observations
    if isfield(prior.noise, 'skip_dtype') && prior.noise.skip_dtype(d)
        continue;
    end

    s = sigma_e(d);

    % hard bounds: reject immediately if gamma_d outside [floor, ceiling]
    if s <= 0 || s < sigma_min(d) || s > sigma_max(d)
        log_p = -Inf;
        return;
    end

    % log-normal prior: Gaussian in log-space centered at log(1.0) = 0
    % this means gamma_d = 1.0 is the most probable value a priori
    % but the wide spread (2.0) makes this nearly uninformative
    log_s = log(s);
    log_mu = log(sigma_prior(d));
    log_p = log_p - 0.5 * ((log_s - log_mu) / log_sigma_width)^2 - log(s);
end

end
