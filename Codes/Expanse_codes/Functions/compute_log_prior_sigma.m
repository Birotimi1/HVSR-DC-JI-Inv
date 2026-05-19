function log_p = compute_log_prior_sigma(sigma_e, prior)
% Log-normal prior on sigma centered at sigma_prior from data uncertainties

log_p = 0;
sigma_prior = prior.noise.sigma_prior;
sigma_min = prior.noise.sigma_min;
sigma_max = prior.noise.sigma_max;

% log-normal width: 1 decade spread around the prior center
log_sigma_width = 1.0;

for d = 1:prior.noise.n_noise
    % skip data types flagged by prior_sigma_from_data
    if isfield(prior.noise, 'skip_dtype') && prior.noise.skip_dtype(d)
        continue;
    end

    s = sigma_e(d);

    % hard bounds check
    if s <= 0 || s < sigma_min(d) || s > sigma_max(d)
        log_p = -Inf;
        return;
    end

    % log-normal prior: Gaussian in log-space centered at log(sigma_prior)
    log_s = log(s);
    log_mu = log(sigma_prior(d));
    log_p = log_p - 0.5 * ((log_s - log_mu) / log_sigma_width)^2 - log(s);
end

end
