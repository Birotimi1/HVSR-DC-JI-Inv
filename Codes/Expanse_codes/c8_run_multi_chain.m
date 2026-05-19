function [results, accept_info_all] = c8_run_multi_chain(data, prior)

n_chains = prior.mcmc.n_chains;
n_params = prior.n_params;
n_noise  = prior.noise.n_noise;
skip_dtype = prior.noise.skip_dtype(:);

if n_chains < 2
    fprintf('[c8] Running single chain\n');
    if isfield(prior.mcmc, 'base_seed') && ~isempty(prior.mcmc.base_seed)
        rng(prior.mcmc.base_seed);
    else
        rng('shuffle');
    end
    [chain_res, chain_ai] = c7_run_mcmc(data, prior);
    results = chain_res;
    results.Rhat = NaN(n_params, 1);
    accept_info_all = {chain_ai};
    return;
end

fprintf('[c8] Running %d chains\n', n_chains);

chain_results = cell(n_chains, 1);
accept_info_all = cell(n_chains, 1);

if isfield(prior.mcmc, 'base_seed') && ~isempty(prior.mcmc.base_seed)
    base_seed = prior.mcmc.base_seed;
else
    base_seed = sum(100*clock);
end

streamType = 'Threefry';
use_parallel = ~isempty(ver('parallel')) && ~isempty(gcp('nocreate'));

if use_parallel
    parfor c = 1:n_chains
        s = RandStream(streamType, 'Seed', base_seed);
        s.Substream = c;
        RandStream.setGlobalStream(s);
        fprintf('[c8] Starting chain %d (parallel)\n', c);
        [chain_results{c}, accept_info_all{c}] = c7_run_mcmc(data, prior);
    end
else
    for c = 1:n_chains
        s = RandStream(streamType, 'Seed', base_seed);
        s.Substream = c;
        RandStream.setGlobalStream(s);
        fprintf('[c8] Starting chain %d (serial)\n', c);
        [chain_results{c}, accept_info_all{c}] = c7_run_mcmc(data, prior);
    end
end

% collate all chains
results = collate_chain_results(chain_results, prior, accept_info_all);

% store per-chain results for downstream access
results.chain_results = chain_results;
results.prior = prior;

% store station info
results.station = struct();
if isfield(data, 'name'), results.station.name = data.name; end
if isfield(data, 'lat'),  results.station.lat  = data.lat;  end
if isfield(data, 'lon'),  results.station.lon  = data.lon;  end
results.data = data;

% total elapsed time across all chains
total_elapsed = 0;
for c = 1:n_chains
    if isfield(chain_results{c}, 'elapsed_seconds')
        total_elapsed = total_elapsed + chain_results{c}.elapsed_seconds;
    end
end
results.total_elapsed_seconds = total_elapsed;
results.total_elapsed_str = sprintf('%dh %dm %.0fs', ...
    floor(total_elapsed/3600), ...
    floor(mod(total_elapsed, 3600)/60), ...
    mod(total_elapsed, 60));

% summary
fprintf('\n[c8] Multi-chain results:\n');
fprintf('  Base seed: %g\n', base_seed);
fprintf('  Chains: %d, Total clean samples: %d\n', n_chains, results.n_total);
fprintf('  MAP log-likelihood: %.2f\n', results.map_logL);
fprintf('  Total elapsed: %s\n', results.total_elapsed_str);

% shake summary
total_shakes = 0;
for c = 1:n_chains
    total_shakes = total_shakes + accept_info_all{c}.shake_count;
end
fprintf('  Total shakes across chains: %d\n', total_shakes);

fprintf('\n[c8] Parameter estimates (mean +/- std):\n');
for i = 1:n_params
    fprintf('  %s: %.4f +/- %.4f %s (Rhat=%.3f)\n', ...
        prior.param_names{i}, results.theta_mean(i), results.theta_std(i), ...
        prior.param_units{i}, results.Rhat(i));
end

fprintf('\n[c8] Noise scale estimates (mean +/- std):\n');
for d = 1:n_noise
    if skip_dtype(d)
        fprintf('  %s: SKIPPED\n', prior.noise.names{d});
    else
        fprintf('  %s: %.4f +/- %.4f\n', prior.noise.names{d}, results.sigma_mean(d), results.sigma_std(d));
    end
end

n_converged = sum(results.Rhat < 1.1);
fprintf('\n[c8] Convergence: %d/%d parameters have Rhat < 1.1\n', n_converged, n_params);

end
