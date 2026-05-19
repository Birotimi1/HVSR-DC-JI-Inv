function empprior = build_empirical_prior(prior, n_samples)
% Direct sampling from the prior to map out the empirical distribution
% of Vs profiles, interface depths, and derived quantities

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

% depth grid for interpolated profiles
dz = 0.001;
zmax = sum(ub(idx_h));
zinterp = (0 : dz : zmax)';
nz = length(zinterp);

% preallocate
empprior.n_samples = n_samples;
empprior.zinterp = zinterp;
empprior.theta = zeros(n_samples, n_params);
empprior.total_depth = zeros(n_samples, 1);
empprior.zsed = zeros(n_samples, 1);
empprior.zmoh = NaN(n_samples, 1);
empprior.vs_profiles = zeros(n_samples, nz);
empprior.layer_depths = zeros(n_samples, n_thick);
empprior.layer_vs = zeros(n_samples, n_layers);

% Moho detection threshold
vs_moho_thresh = 4.0;

% sediment detection: largest velocity jump in top 5 layers where vs < 2.0 transitions to vs >= 2.0
n_sed_search = 5;

fprintf('[empprior] Sampling %d models from prior\n', n_samples);
tic_start = tic;

count = 0;
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

    % store raw parameters
    empprior.theta(count, :) = theta';

    % extract h and vs
    h = theta(idx_h);
    vs = theta(idx_vs);
    empprior.layer_vs(count, :) = vs';

    % interface depths
    z_interfaces = cumsum(h);
    empprior.layer_depths(count, :) = z_interfaces';
    empprior.total_depth(count) = sum(h);

    % build model matrix and interpolate
    model_matrix = zeros(n_layers, 4);
    for il = 1:n_layers
        if il <= n_thick
            model_matrix(il, 1) = h(il);
        else
            model_matrix(il, 1) = 0;
        end
        model_matrix(il, 3) = vs(il);
    end

    % build staircase profile on zinterp
    z_top = [0; z_interfaces];
    vs_profile = zeros(nz, 1);
    for il = 1:n_layers
        zt = z_top(il);
        if il < n_layers
            zb = z_top(il + 1);
        else
            zb = zmax;
        end
        mask = zinterp >= zt & zinterp < zb;
        vs_profile(mask) = vs(il);
    end
    vs_profile(end) = vs(end);
    empprior.vs_profiles(count, :) = vs_profile';

    % sediment detection
    dv = diff(vs);
    sed_candidates = find(vs(1:n_sed_search) < 2.0 & vs(2:n_sed_search+1) >= 2.0 & dv(1:n_sed_search) > 0);
    if ~isempty(sed_candidates)
        [~, ibest] = max(dv(sed_candidates));
        empprior.zsed(count) = z_interfaces(sed_candidates(ibest));
    else
        empprior.zsed(count) = NaN;
    end

    % Moho detection: shallowest depth where vs >= 4.0
    iz_moho = find(vs >= vs_moho_thresh, 1, 'first');
    if ~isempty(iz_moho)
        if iz_moho == 1
            empprior.zmoh(count) = 0;
        else
            empprior.zmoh(count) = z_interfaces(iz_moho - 1);
        end
    end

    % progress
    if mod(count, 20000) == 0
        fprintf('[empprior] %d/%d samples (%.0f attempts, %.1f%% acceptance)\n', ...
            count, n_samples, attempts, 100*count/attempts);
    end
end

elapsed = toc(tic_start);
acceptance_rate = n_samples / attempts;
fprintf('[empprior] Complete: %d samples, %d attempts, %.1f%% acceptance, %.1f s\n', ...
    n_samples, attempts, 100*acceptance_rate, elapsed);

% summary statistics
empprior.acceptance_rate = acceptance_rate;
empprior.attempts = attempts;

% depth percentiles
empprior.vs_median = median(empprior.vs_profiles, 1)';
empprior.vs_mean   = mean(empprior.vs_profiles, 1)';
empprior.vs_q05    = prctile(empprior.vs_profiles, 5, 1)';
empprior.vs_q16    = prctile(empprior.vs_profiles, 16, 1)';
empprior.vs_q84    = prctile(empprior.vs_profiles, 84, 1)';
empprior.vs_q95    = prctile(empprior.vs_profiles, 95, 1)';

% interface statistics
empprior.zsed_median = nanmedian(empprior.zsed);
empprior.zsed_mean   = nanmean(empprior.zsed);
empprior.zmoh_median = nanmedian(empprior.zmoh);
empprior.zmoh_mean   = nanmean(empprior.zmoh);

% per-layer statistics
empprior.h_median = median(empprior.theta(:, idx_h), 1)';
empprior.h_mean   = mean(empprior.theta(:, idx_h), 1)';
empprior.vs_layer_median = median(empprior.layer_vs, 1)';
empprior.vs_layer_mean   = mean(empprior.layer_vs, 1)';

fprintf('[empprior] Sediment depth: median=%.3f km, mean=%.3f km\n', empprior.zsed_median, empprior.zsed_mean);
fprintf('[empprior] Moho depth: median=%.1f km, mean=%.1f km\n', empprior.zmoh_median, empprior.zmoh_mean);
fprintf('[empprior] Total depth range: %.1f to %.1f km\n', min(empprior.total_depth), max(empprior.total_depth));

end
