cat << 'EOF' > /expanse/lustre/projects/syu127/brotimi/HVSR-Joint-Inversion/Codes/Mod5_HBI/functions_hbi_new/c0_build_empirical_prior.m
function empprior = c0_build_empirical_prior(prior, n_samples)
% Direct sampling from the prior to map the empirical distribution
% Computes running statistics to avoid storing full profile matrix

if nargin < 2 || isempty(n_samples)
    n_samples = 100000;
end

n_params = prior.n_params;
idx_h = prior.idx.h;
idx_vs = prior.idx.vs;
n_layers = length(idx_vs);
n_thick = length(idx_h);
lb = prior.bounds.lower(:);
ub = prior.bounds.upper(:);

% coarse depth grid for statistics (not 1m spacing)
dz = 0.05;
zmax = sum(ub(idx_h));
zcoarse = (0 : dz : zmax)';
nz = length(zcoarse);

% preallocate storage for parameters only
empprior.n_samples = n_samples;
empprior.zcoarse = zcoarse;
empprior.theta = zeros(n_samples, n_params);
empprior.total_depth = zeros(n_samples, 1);
empprior.zsed = NaN(n_samples, 1);
empprior.zmoh = NaN(n_samples, 1);

% running statistics for Vs profiles (Welford online algorithm)
vs_sum = zeros(nz, 1);
vs_sum2 = zeros(nz, 1);
vs_sorted_sub = zeros(min(n_samples, 10000), nz);
store_every = max(1, floor(n_samples / 10000));

vs_moho_thresh = 4.0;
n_sed_search = 5;

fprintf('[c0] Sampling %d models from prior (coarse grid dz=%.3f km, nz=%d)\n', n_samples, dz, nz);
tic_start = tic;

count = 0;
sub_count = 0;
attempts = 0;

while count < n_samples
    attempts = attempts + 1;

    % draw random theta within bounds
    theta = lb + (ub - lb) .* rand(n_params, 1);

    % enforce monotonicity on Vs
    vs = theta(idx_vs);
    if ~issorted(vs)
        vs = sort(vs);
        theta(idx_vs) = vs;
    end

    % verify feasibility
    [is_feasible, ~] = c4_check_feasibility_internal(theta, prior);
    if ~is_feasible
        continue;
    end

    count = count + 1;
    h = theta(idx_h);
    empprior.theta(count, :) = theta';
    empprior.total_depth(count) = sum(h);

    % build staircase on coarse grid
    z_interfaces = cumsum(h);
    z_top = [0; z_interfaces];
    vs_profile = zeros(nz, 1);
    for il = 1:n_layers
        zt = z_top(il);
        if il < n_layers
            zb = z_top(il + 1);
        else
            zb = zmax;
        end
        mask = zcoarse >= zt & zcoarse < zb;
        vs_profile(mask) = vs(il);
    end
    vs_profile(end) = vs(end);

    % running statistics
    vs_sum = vs_sum + vs_profile;
    vs_sum2 = vs_sum2 + vs_profile.^2;

    % subsample for percentile computation
    if mod(count, store_every) == 0
        sub_count = sub_count + 1;
        vs_sorted_sub(sub_count, :) = vs_profile';
    end

    % sediment detection
    dv = diff(vs);
    sed_candidates = find(vs(1:n_sed_search) < 2.0 & vs(2:n_sed_search+1) >= 2.0 & dv(1:n_sed_search) > 0);
    if ~isempty(sed_candidates)
        [~, ibest] = max(dv(sed_candidates));
        empprior.zsed(count) = z_interfaces(sed_candidates(ibest));
    end

    % Moho detection
    iz_moho = find(vs >= vs_moho_thresh, 1, 'first');
    if ~isempty(iz_moho)
        if iz_moho == 1
            empprior.zmoh(count) = 0;
        else
            empprior.zmoh(count) = z_interfaces(iz_moho - 1);
        end
    end

    if mod(count, 20000) == 0
        fprintf('[c0] %d/%d samples (%.0f attempts, %.1f%% acceptance)\n', ...
            count, n_samples, attempts, 100*count/attempts);
    end
end

elapsed = toc(tic_start);
acceptance_rate = n_samples / attempts;
fprintf('[c0] Complete: %d samples, %d attempts, %.1f%% acceptance, %.1f s\n', ...
    n_samples, attempts, 100*acceptance_rate, elapsed);

% compute statistics from running sums
empprior.vs_mean = vs_sum / n_samples;
empprior.vs_std = sqrt(vs_sum2/n_samples - empprior.vs_mean.^2);

% compute percentiles from subsampled profiles
vs_sorted_sub = vs_sorted_sub(1:sub_count, :);
empprior.vs_median = median(vs_sorted_sub, 1)';
empprior.vs_q05 = prctile(vs_sorted_sub, 5, 1)';
empprior.vs_q16 = prctile(vs_sorted_sub, 16, 1)';
empprior.vs_q84 = prctile(vs_sorted_sub, 84, 1)';
empprior.vs_q95 = prctile(vs_sorted_sub, 95, 1)';

% interface statistics
empprior.zsed_median = nanmedian(empprior.zsed);
empprior.zsed_mean = nanmean(empprior.zsed);
empprior.zmoh_median = nanmedian(empprior.zmoh);
empprior.zmoh_mean = nanmean(empprior.zmoh);

% per-layer statistics
empprior.h_median = median(empprior.theta(:, idx_h), 1)';
empprior.h_mean = mean(empprior.theta(:, idx_h), 1)';
empprior.vs_layer_median = median(empprior.theta(:, idx_vs), 1)';
empprior.vs_layer_mean = mean(empprior.theta(:, idx_vs), 1)';

empprior.acceptance_rate = acceptance_rate;
empprior.attempts = attempts;

fprintf('[c0] Sediment depth: median=%.3f km, mean=%.3f km\n', empprior.zsed_median, empprior.zsed_mean);
fprintf('[c0] Moho depth: median=%.1f km, mean=%.1f km\n', empprior.zmoh_median, empprior.zmoh_mean);
fprintf('[c0] Total depth range: %.1f to %.1f km\n', min(empprior.total_depth), max(empprior.total_depth));

end
EOF
