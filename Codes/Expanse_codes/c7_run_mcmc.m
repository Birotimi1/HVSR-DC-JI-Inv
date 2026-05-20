function [results, accept_info] = c7_run_mcmc(data, prior)

n_iter   = prior.mcmc.n_iterations;
burn_in  = prior.mcmc.burn_in;
thin     = prior.mcmc.thin;
n_params = prior.n_params;
n_noise  = prior.noise.n_noise;

% correlated noise setup
use_corr = isfield(prior.noise, 'use_correlated') && prior.noise.use_correlated;
if use_corr
    n_corrlen = n_noise;
    n_total_params = n_params + n_noise + n_corrlen;
else
    n_corrlen = 0;
    n_total_params = n_params + n_noise;
end

% sample storage
n_saved = floor((n_iter - burn_in) / thin);
if n_saved < 1
    error('c7_run_mcmc: No samples will be saved.');
end

samples_theta = zeros(n_saved, n_params);
samples_sigma = zeros(n_saved, n_noise);
samples_logL  = zeros(n_saved, 1);
if use_corr
    samples_corrlen = zeros(n_saved, n_noise);
end

% prediction sizes
n_hvsr = 0; n_ellip = 0; n_cph = 0; n_ugr = 0;
if isfield(data, 'hvsr')  && isfield(data.hvsr, 'f'),  n_hvsr  = length(data.hvsr.f);  end
if isfield(data, 'ellip') && isfield(data.ellip, 'T'), n_ellip = length(data.ellip.T); end
if isfield(data, 'cph')   && isfield(data.cph, 'T'),   n_cph   = length(data.cph.T);   end
if isfield(data, 'ugr')   && isfield(data.ugr, 'T'),   n_ugr   = length(data.ugr.T);   end

samples_pred_hvsr  = NaN(n_saved, n_hvsr);
samples_pred_ellip = NaN(n_saved, n_ellip);
samples_pred_cph   = NaN(n_saved, n_cph);
samples_pred_ugr   = NaN(n_saved, n_ugr);

% initialize model
model   = c2_initialize_hbi_model(prior);
theta   = model.theta(:);
sigma_e = model.sigma_e(:);

% correlation length setup
if use_corr
    corrlen = prior.noise.corrlen_init(:);
    corrlen_step = prior.noise.corrlen_step(:);
    corrlen_lb = prior.noise.corrlen_lb(:);
    corrlen_ub = prior.noise.corrlen_ub(:);
else
    corrlen = [];
end

% sigma proposal step sizes and bounds from prior
sigma_step = [0.05; 0.05; 0.05; 0.05];
sigma_lb = prior.noise.sigma_min(:);
sigma_ub = prior.noise.sigma_max(:);
skip_dtype = prior.noise.skip_dtype(:);

% temperature annealing parameters
tau_max = prior.mcmc.tau_max;
tau_cooldown = prior.mcmc.tau_cooldown;

% initial likelihood
[log_L, ~, pred, Phi, N_d, chol_cache] = c3_compute_likelihood(theta, sigma_e, data, prior, corrlen);
log_prior_sigma = compute_log_prior_sigma(sigma_e, prior);
if ~isfinite(log_L)
    log_L = -1e10;
end

% initialize tracking structure
accept_info = make_accept_info(n_iter, n_saved, n_params, n_noise);

% best model tracking
best = struct('theta', theta, 'sigma', sigma_e, 'logL', log_L);
if use_corr
    best.corrlen = corrlen;
end

sample_idx = 0;

fprintf('[c7] Starting MCMC: %d iterations, burn_in=%d, thin=%d\n', n_iter, burn_in, thin);
fprintf('[c7] Tau annealing: tau_max=%.1f, cooldown=%d\n', tau_max, tau_cooldown);
if use_corr
    fprintf('[c7] Correlated noise: ON\n');
end
fprintf('[c7] Delayed rejection: ON (neighbor-aware)\n');
fprintf('[c7] Initial log_L = %.4f\n', log_L);

tic_start = tic;

for iter = 1:n_iter

    % compute temperature tau
    if ~accept_info.in_shake
        % burn-in annealing: decays from tau_max to 1 over tau_cooldown
        tau = 1 + (tau_max - 1) * erfc(iter / (tau_cooldown / 3));
    else
        % shake cooldown: decays from shake_tau_max to 1
        shake_elapsed = iter - accept_info.shake_iter_start;
        tau = 1 + (accept_info.shake_tau_max - 1) * erfc(shake_elapsed / (accept_info.shake_decay_rate / 3));
        if tau < accept_info.shake_tau_settled
            tau = 1.0;
            accept_info.in_shake = false;
            fprintf('[c7] Shake %d cooldown complete at iter %d\n', accept_info.shake_count, iter);
        end
    end

    % record tau and log_L
    accept_info.tau(iter) = tau;
    accept_info.log_L(iter) = log_L;

    % random parameter selection
    j = randi(n_total_params);
    accept_info.param_perturbed(iter) = j;

    if j <= n_params
        % theta perturbation via delayed rejection
        accept_info.total_count_theta(j) = accept_info.total_count_theta(j) + 1;

        [theta_new, log_L_new, pred_new, Phi_new, N_d_new, was_accepted, dr_stage] = ...
            delayed_rejection(theta, log_L, sigma_e, data, prior, corrlen, chol_cache, j, tau);

        accept_info.dr_stage(iter) = dr_stage;
        if was_accepted
            theta = theta_new;
            log_L = log_L_new;
            pred  = pred_new;
            Phi   = Phi_new;
            N_d   = N_d_new;
            accept_info.accept_count_theta(j) = accept_info.accept_count_theta(j) + 1;
            accept_info.accepted(iter) = true;
        end

    elseif j <= n_params + n_noise
        % sigma perturbation
        d = j - n_params;
        accept_info.total_count_sigma(d) = accept_info.total_count_sigma(d) + 1;

        % skip if this data type has no valid uncertainties
        if skip_dtype(d)
            accept_info.accepted(iter) = false;
        else
            log_sigma_prop = log(sigma_e(d)) + sigma_step(d) * tau * randn;
            sigma_prop = exp(log_sigma_prop);

            if sigma_prop >= sigma_lb(d) && sigma_prop <= sigma_ub(d)
                sigma_e_prop = sigma_e;
                sigma_e_prop(d) = sigma_prop;

                [log_L_prop, ~, pred_prop, Phi_prop, N_d_prop, chol_cache_prop] = ...
                    c3_compute_likelihood(theta, sigma_e_prop, data, prior, corrlen);
                log_prior_sigma_prop = compute_log_prior_sigma(sigma_e_prop, prior);

                if isfinite(log_L_prop)
                    log_alpha = (log_L_prop - log_L) ...
                              + (log_prior_sigma_prop - log_prior_sigma) ...
                              + (log_sigma_prop - log(sigma_e(d))) ...
                              + log(tau);

                    if log(rand()) < log_alpha
                        sigma_e = sigma_e_prop;
                        log_L = log_L_prop;
                        pred  = pred_prop;
                        Phi   = Phi_prop;
                        N_d   = N_d_prop;
                        log_prior_sigma = log_prior_sigma_prop;
                        chol_cache = chol_cache_prop;
                        accept_info.accept_count_sigma(d) = accept_info.accept_count_sigma(d) + 1;
                        accept_info.accepted(iter) = true;
                    end
                end
            end
        end

    else
        % corrlen perturbation
        d = j - n_params - n_noise;
        accept_info.total_count_corrlen(d) = accept_info.total_count_corrlen(d) + 1;

        % skip if this data type has no valid uncertainties
        if skip_dtype(d)
            accept_info.accepted(iter) = false;
        else
            log_L_prop_val = log(corrlen(d)) + corrlen_step(d) * tau * randn;
            corrlen_prop_val = exp(log_L_prop_val);

            if corrlen_prop_val >= corrlen_lb(d) && corrlen_prop_val <= corrlen_ub(d)
                corrlen_prop = corrlen;
                corrlen_prop(d) = corrlen_prop_val;

                [log_L_prop, ~, pred_prop, Phi_prop, N_d_prop, chol_cache_prop] = ...
                    c3_compute_likelihood(theta, sigma_e, data, prior, corrlen_prop);

                if isfinite(log_L_prop)
                    log_alpha = (log_L_prop - log_L) ...
                              + (log_L_prop_val - log(corrlen(d))) ...
                              + log(tau);

                    if log(rand()) < log_alpha
                        corrlen = corrlen_prop;
                        log_L = log_L_prop;
                        pred  = pred_prop;
                        Phi   = Phi_prop;
                        N_d   = N_d_prop;
                        chol_cache = chol_cache_prop;
                        accept_info.accept_count_corrlen(d) = accept_info.accept_count_corrlen(d) + 1;
                        accept_info.accepted(iter) = true;
                    end
                end
            end
        end
    end

    % track best model
    if log_L > best.logL
        best.theta = theta;
        best.sigma = sigma_e;
        best.logL  = log_L;
        if use_corr
            best.corrlen = corrlen;
        end
    end

    % compute baseline std at end of burn-in for shake detection
    if iter == burn_in
        baseline_window = accept_info.log_L(max(1, burn_in-2000):burn_in);
        baseline_window = baseline_window(isfinite(baseline_window));
        if length(baseline_window) > 10
            accept_info.baseline_std = std(baseline_window);
            accept_info.shake_std_thresh = accept_info.baseline_std * accept_info.shake_std_fraction;
        else
            accept_info.shake_std_thresh = Inf;
        end
        fprintf('[c7] Burn-in complete. baseline_std=%.4f, shake_thresh=%.4f\n', ...
            accept_info.baseline_std, accept_info.shake_std_thresh);
    end

    % shake detection post burn-in
    if iter > burn_in && ~accept_info.in_shake
        % fill rolling buffer
        accept_info.buffer_idx = mod(accept_info.buffer_idx, accept_info.shake_window) + 1;
        accept_info.log_L_buffer(accept_info.buffer_idx) = log_L;

        % check for stagnation every shake_window iterations
        if mod(iter - burn_in, accept_info.shake_window) == 0 ...
                && accept_info.shake_count < accept_info.max_shakes
            buf = accept_info.log_L_buffer(isfinite(accept_info.log_L_buffer));
            if length(buf) >= accept_info.shake_window * 0.8
                current_std = std(buf);
                if current_std < accept_info.shake_std_thresh
                    accept_info.in_shake = true;
                    accept_info.shake_count = accept_info.shake_count + 1;
                    accept_info.shake_iter_start = iter;
                    accept_info.log_L_buffer(:) = NaN;
                    accept_info.buffer_idx = 0;

                    % retroactively discard samples from the stagnation window
                    % these were collected while the chain was stuck
                    stag_start_iter = iter - accept_info.shake_window;
                    for ks = 1:sample_idx
                        sample_iter = burn_in + ks * thin;
                        if sample_iter > stag_start_iter && sample_iter <= iter
                            accept_info.sample_valid(ks) = false;
                        end
                    end

                    fprintf('[c7] Shake %d triggered at iter %d (std=%.4f < thresh=%.4f), discarded %d stagnation samples\n', ...
                        accept_info.shake_count, iter, current_std, accept_info.shake_std_thresh, ...
                        sum(~accept_info.sample_valid(1:sample_idx)));
                end
            end
        end
    end

    % mark shake iterations
    if accept_info.in_shake
        accept_info.is_shake(iter) = true;
    end

    % store post burn-in samples
    if iter > burn_in && mod(iter - burn_in, thin) == 0
        sample_idx = sample_idx + 1;
        samples_theta(sample_idx, :) = theta';
        samples_sigma(sample_idx, :) = sigma_e';
        samples_logL(sample_idx)     = log_L;
        if use_corr
            samples_corrlen(sample_idx, :) = corrlen';
        end

        % flag sample validity based on shake state
        if accept_info.in_shake
            accept_info.sample_valid(sample_idx) = false;
        end

        % store predictions
        if ~isempty(pred.hvsr) && n_hvsr > 0
            pv = pred.hvsr(:);
            nc = min(length(pv), n_hvsr);
            samples_pred_hvsr(sample_idx, 1:nc) = pv(1:nc)';
        end
        if ~isempty(pred.ellip) && n_ellip > 0
            pv = pred.ellip(:);
            nc = min(length(pv), n_ellip);
            samples_pred_ellip(sample_idx, 1:nc) = pv(1:nc)';
        end
        if ~isempty(pred.cph) && n_cph > 0
            pv = pred.cph(:);
            nc = min(length(pv), n_cph);
            samples_pred_cph(sample_idx, 1:nc) = pv(1:nc)';
        end
        if ~isempty(pred.ugr) && n_ugr > 0
            pv = pred.ugr(:);
            nc = min(length(pv), n_ugr);
            samples_pred_ugr(sample_idx, 1:nc) = pv(1:nc)';
        end
    end

    % progress reporting
    if mod(iter, 2000) == 0
        acc_theta = sum(accept_info.accept_count_theta) / max(sum(accept_info.total_count_theta), 1);
        acc_sigma = sum(accept_info.accept_count_sigma) / max(sum(accept_info.total_count_sigma), 1);
        sigma_str = sprintf('%.3f ', sigma_e);
        if use_corr
            acc_corrlen = sum(accept_info.accept_count_corrlen) / max(sum(accept_info.total_count_corrlen), 1);
            corrlen_str = sprintf('%.3f ', corrlen);
            fprintf('[c7] Iter %d/%d, logL=%.2f, tau=%.2f, acc_th=%.3f, acc_sig=%.3f, acc_L=%.3f, sigma=[%s], L=[%s]\n', ...
                iter, n_iter, log_L, tau, acc_theta, acc_sigma, acc_corrlen, strtrim(sigma_str), strtrim(corrlen_str));
        else
            fprintf('[c7] Iter %d/%d, logL=%.2f, tau=%.2f, acc_th=%.3f, acc_sig=%.3f, sigma=[%s]\n', ...
                iter, n_iter, log_L, tau, acc_theta, acc_sigma, strtrim(sigma_str));
        end
    end
end

% trim sample arrays
samples_theta = samples_theta(1:sample_idx, :);
samples_sigma = samples_sigma(1:sample_idx, :);
samples_logL  = samples_logL(1:sample_idx);
samples_pred_hvsr  = samples_pred_hvsr(1:sample_idx, :);
samples_pred_ellip = samples_pred_ellip(1:sample_idx, :);
samples_pred_cph   = samples_pred_cph(1:sample_idx, :);
samples_pred_ugr   = samples_pred_ugr(1:sample_idx, :);
accept_info.sample_valid = accept_info.sample_valid(1:sample_idx);

% pack results
results.samples_theta = samples_theta;
results.samples_sigma = samples_sigma;
results.samples_logL  = samples_logL;
results.sample_valid  = accept_info.sample_valid;
results.samples_pred_hvsr  = samples_pred_hvsr;
results.samples_pred_ellip = samples_pred_ellip;
results.samples_pred_cph   = samples_pred_cph;
results.samples_pred_ugr   = samples_pred_ugr;
results.accept_count    = accept_info.accept_count_theta;
results.total_count     = accept_info.total_count_theta;
results.acceptance_rate = accept_info.accept_count_theta ./ max(accept_info.total_count_theta, 1);
results.accept_count_sigma = accept_info.accept_count_sigma;
results.total_count_sigma  = accept_info.total_count_sigma;
results.acceptance_rate_sigma = accept_info.accept_count_sigma ./ max(accept_info.total_count_sigma, 1);
results.final_theta = theta;
results.final_sigma = sigma_e;
results.final_logL  = log_L;
results.best_theta = best.theta;
results.best_sigma = best.sigma;
results.best_logL  = best.logL;
results.prior     = prior;
results.n_samples = sample_idx;

if use_corr
    samples_corrlen = samples_corrlen(1:sample_idx, :);
    results.samples_corrlen = samples_corrlen;
    results.accept_count_corrlen = accept_info.accept_count_corrlen;
    results.total_count_corrlen  = accept_info.total_count_corrlen;
    results.acceptance_rate_corrlen = accept_info.accept_count_corrlen ./ max(accept_info.total_count_corrlen, 1);
    results.final_corrlen = corrlen;
    results.best_corrlen  = best.corrlen;
end

% Gelman-Rubin from single chain (placeholder, computed properly in collate)
results.Rhat = NaN(n_params, 1);

% station and data info
results.station = struct();
if isfield(data, 'name'), results.station.name = data.name; end
if isfield(data, 'lat'),  results.station.lat  = data.lat;  end
if isfield(data, 'lon'),  results.station.lon  = data.lon;  end
results.data = data;
results.elapsed_seconds = toc(tic_start);
results.elapsed_str = sprintf('%dh %dm %.0fs', ...
    floor(results.elapsed_seconds/3600), ...
    floor(mod(results.elapsed_seconds, 3600)/60), ...
    mod(results.elapsed_seconds, 60));

% summary
n_shakes = accept_info.shake_count;
n_valid = sum(accept_info.sample_valid);
fprintf('[c7] MCMC complete. %d samples saved, %d valid, %d shakes\n', sample_idx, n_valid, n_shakes);
fprintf('[c7] Per-parameter acceptance rates (theta):\n');
for i = 1:n_params
    fprintf('  %s: %.3f (%d/%d)\n', prior.param_names{i}, results.acceptance_rate(i), ...
        accept_info.accept_count_theta(i), accept_info.total_count_theta(i));
end
fprintf('[c7] Per-noise acceptance rates (sigma):\n');
for d = 1:n_noise
    if skip_dtype(d)
        fprintf('  %s: SKIPPED\n', prior.noise.names{d});
    else
        fprintf('  %s: %.3f (%d/%d)\n', prior.noise.names{d}, results.acceptance_rate_sigma(d), ...
            accept_info.accept_count_sigma(d), accept_info.total_count_sigma(d));
    end
end
if use_corr
    fprintf('[c7] Per-noise acceptance rates (corrlen):\n');
    for d = 1:n_noise
        if skip_dtype(d)
            fprintf('  %s: SKIPPED\n', prior.noise.corrlen_names{d});
        else
            fprintf('  %s: %.3f (%d/%d)\n', prior.noise.corrlen_names{d}, results.acceptance_rate_corrlen(d), ...
                accept_info.accept_count_corrlen(d), accept_info.total_count_corrlen(d));
        end
    end
end
fprintf('[c7] Best logL = %.3f\n', results.best_logL);

end
