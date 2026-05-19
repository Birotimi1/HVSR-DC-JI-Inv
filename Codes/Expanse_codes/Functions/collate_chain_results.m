function collated = collate_chain_results(chain_results, prior, accept_info_all)
% Collate post-burn-in samples from all chains into a single structure
% chain_results: cell array of per-chain result structs from c7
% prior: prior struct with mcmc settings
% accept_info_all: cell array of per-chain accept_info structs (optional)

n_chains = length(chain_results);
burn_in = prior.mcmc.burn_in;
thin = prior.mcmc.thin;
n_params = prior.n_params;
n_noise = prior.noise.n_noise;

% count valid samples across all chains
total_samples = 0;
for ic = 1:n_chains
    if isempty(chain_results{ic}), continue; end
    total_samples = total_samples + chain_results{ic}.n_samples;
end

if total_samples == 0
    error('collate_chain_results: no samples found across %d chains', n_chains);
end

% determine prediction sizes from first non-empty chain
for ic = 1:n_chains
    if ~isempty(chain_results{ic})
        n_hvsr  = size(chain_results{ic}.samples_pred_hvsr, 2);
        n_ellip = size(chain_results{ic}.samples_pred_ellip, 2);
        n_cph   = size(chain_results{ic}.samples_pred_cph, 2);
        n_ugr   = size(chain_results{ic}.samples_pred_ugr, 2);
        break;
    end
end

% preallocate
collated.samples_theta = zeros(total_samples, n_params);
collated.samples_sigma = zeros(total_samples, n_noise);
collated.samples_logL  = zeros(total_samples, 1);
collated.chain_id      = zeros(total_samples, 1);
collated.sample_valid  = true(total_samples, 1);
collated.samples_pred_hvsr  = NaN(total_samples, n_hvsr);
collated.samples_pred_ellip = NaN(total_samples, n_ellip);
collated.samples_pred_cph   = NaN(total_samples, n_cph);
collated.samples_pred_ugr   = NaN(total_samples, n_ugr);

if isfield(chain_results{1}, 'samples_corrlen')
    collated.samples_corrlen = zeros(total_samples, n_noise);
    has_corrlen = true;
else
    has_corrlen = false;
end

% fill from each chain
idx = 0;
for ic = 1:n_chains
    cr = chain_results{ic};
    if isempty(cr), continue; end

    ns = cr.n_samples;
    if ns == 0, continue; end
    rows = idx + (1:ns);

    collated.samples_theta(rows, :) = cr.samples_theta;
    collated.samples_sigma(rows, :) = cr.samples_sigma;
    collated.samples_logL(rows)     = cr.samples_logL;
    collated.chain_id(rows)         = ic;

    collated.samples_pred_hvsr(rows, :)  = cr.samples_pred_hvsr;
    collated.samples_pred_ellip(rows, :) = cr.samples_pred_ellip;
    collated.samples_pred_cph(rows, :)   = cr.samples_pred_cph;
    collated.samples_pred_ugr(rows, :)   = cr.samples_pred_ugr;

    if has_corrlen
        collated.samples_corrlen(rows, :) = cr.samples_corrlen;
    end

    % mark invalid samples from shake periods
    if nargin >= 3 && ~isempty(accept_info_all) && ~isempty(accept_info_all{ic})
        collated.sample_valid(rows) = accept_info_all{ic}.sample_valid(1:ns);
    end

    idx = idx + ns;
end

% trim if any chains were empty
collated.samples_theta = collated.samples_theta(1:idx, :);
collated.samples_sigma = collated.samples_sigma(1:idx, :);
collated.samples_logL  = collated.samples_logL(1:idx);
collated.chain_id      = collated.chain_id(1:idx);
collated.sample_valid  = collated.sample_valid(1:idx);
collated.samples_pred_hvsr  = collated.samples_pred_hvsr(1:idx, :);
collated.samples_pred_ellip = collated.samples_pred_ellip(1:idx, :);
collated.samples_pred_cph   = collated.samples_pred_cph(1:idx, :);
collated.samples_pred_ugr   = collated.samples_pred_ugr(1:idx, :);
if has_corrlen
    collated.samples_corrlen = collated.samples_corrlen(1:idx, :);
end

% remove shake-flagged samples
valid = collated.sample_valid;
n_discarded = sum(~valid);
if n_discarded > 0
    fprintf('[collate] Discarding %d shake samples out of %d total\n', n_discarded, idx);
end

collated.samples_theta_clean = collated.samples_theta(valid, :);
collated.samples_sigma_clean = collated.samples_sigma(valid, :);
collated.samples_logL_clean  = collated.samples_logL(valid);
collated.chain_id_clean      = collated.chain_id(valid);

% summary statistics from clean samples only
collated.n_total = sum(valid);
collated.n_chains = n_chains;
collated.theta_mean = mean(collated.samples_theta_clean, 1)';
collated.theta_std  = std(collated.samples_theta_clean, 0, 1)';
collated.theta_median = median(collated.samples_theta_clean, 1)';
collated.theta_q05  = prctile(collated.samples_theta_clean, 5, 1)';
collated.theta_q16  = prctile(collated.samples_theta_clean, 16, 1)';
collated.theta_q84  = prctile(collated.samples_theta_clean, 84, 1)';
collated.theta_q95  = prctile(collated.samples_theta_clean, 95, 1)';
collated.sigma_mean = mean(collated.samples_sigma_clean, 1)';
collated.sigma_std  = std(collated.samples_sigma_clean, 0, 1)';

% MAP estimate
[collated.map_logL, imap] = max(collated.samples_logL_clean);
collated.theta_map = collated.samples_theta_clean(imap, :)';
collated.sigma_map = collated.samples_sigma_clean(imap, :)';

% Gelman-Rubin Rhat per parameter from clean samples only
collated.Rhat = compute_rhat(chain_results, accept_info_all, n_params);

fprintf('[collate] %d chains, %d clean samples, best logL = %.2f\n', ...
    n_chains, collated.n_total, collated.map_logL);

end


function Rhat = compute_rhat(chain_results, accept_info_all, n_params)
% Gelman-Rubin convergence diagnostic across chains using clean samples only

n_chains = length(chain_results);
Rhat = NaN(n_params, 1);

chain_means = [];
chain_vars  = [];
chain_ns    = [];

for ic = 1:n_chains
    if isempty(chain_results{ic}), continue; end
    theta = chain_results{ic}.samples_theta;
    ns = size(theta, 1);
    if ns < 2, continue; end

    % exclude shake samples if accept_info available
    if ~isempty(accept_info_all) && length(accept_info_all) >= ic && ~isempty(accept_info_all{ic})
        sv = accept_info_all{ic}.sample_valid(1:ns);
        theta = theta(sv, :);
        ns = size(theta, 1);
        if ns < 2, continue; end
    end

    chain_means = [chain_means; mean(theta, 1)];
    chain_vars  = [chain_vars; var(theta, 0, 1)];
    chain_ns    = [chain_ns; ns];
end

m = size(chain_means, 1);
if m < 2, return; end

n_avg = mean(chain_ns);
grand_mean = mean(chain_means, 1);

for j = 1:n_params
    B = (n_avg / (m - 1)) * sum((chain_means(:, j) - grand_mean(j)).^2);
    W = mean(chain_vars(:, j));
    if W < 1e-20, continue; end
    var_hat = ((n_avg - 1) / n_avg) * W + (1 / n_avg) * B;
    Rhat(j) = sqrt(var_hat / W);
end

end
