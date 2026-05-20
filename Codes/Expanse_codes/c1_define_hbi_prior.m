function prior = c1_define_hbi_prior(data)
%
% Define prior for 10-layer HBI joint inversion
%
% SIGMA / NOISE MODEL:
%   The misfit at each data point i for data type d is:
%
%       x_i = (d_obs_i - d_pred_i)^2 / (gamma_d * sigma_i)^2
%
%   where:
%       sigma_i = per-point measurement uncertainty (fixed, from data)
%       gamma_d = hierarchical noise multiplier (sampled during MCMC)
%       gamma_d * sigma_i = total effective uncertainty at point i
%
%   gamma_d = 1.0 means trust data errors as given
%   gamma_d < 1.0 means data errors are overestimated (squeeze)
%   gamma_d > 1.0 means data errors are underestimated (expand)
%
%   In the code, gamma_d is stored as sigma_e(d).
%   The floor on gamma_d prevents any single data type from dominating.
%   The ceiling is generous to absorb forward modeling error.
%
% INPUT:
%   data = struct with fields hvsr, ellip, cph, ugr (from station datapack)

prior.n_params = 19;
prior.param_names = {'h1','h2','h3','h4','h5','h6','h7','h8','h9', ...
                     'Vs1','Vs2','Vs3','Vs4','Vs5','Vs6','Vs7','Vs8','Vs9','Vs10'};
prior.param_units = {'km','km','km','km','km','km','km','km','km', ...
                     'km/s','km/s','km/s','km/s','km/s','km/s','km/s','km/s','km/s','km/s'};

% parameter index mapping
prior.idx.h  = 1:9;
prior.idx.vs = 10:19;

% monotonicity constraint on Vs (Vs must increase or stay equal with depth)
prior.enforce_monotonicity = true;
prior.monotonic_start_idx  = 10;

% layer thickness bounds [km]
h_min = [0.005; 0.02; 0.08; 0.30; 1.50; 2.00; 4.00; 6.00; 8.00];
h_max = [0.080; 0.30; 1.50; 4.00; 10.0; 15.0; 18.0; 22.0; 28.0];

% shear velocity bounds per layer [km/s]
vs_min = [0.12; 0.25; 0.50; 0.80; 2.00; 2.60; 3.00; 3.40; 3.80; 4.10];
vs_max = [0.90; 1.60; 2.20; 3.20; 3.80; 4.00; 4.30; 4.60; 4.90; 5.20];

prior.bounds.lower = [h_min; vs_min];
prior.bounds.upper = [h_max; vs_max];
prior.depth_min_km = sum(h_min);
prior.depth_max_km = sum(h_max);

% noise model: 4 data types, each with a hierarchical multiplier gamma_d
prior.noise.n_noise = 4;
prior.noise.names = {'gamma_hvsr', 'gamma_ellip', 'gamma_cph', 'gamma_ugr'};

% data-informed sigma setup
% sigma_prior = prior center for gamma_d (1.0 = trust data errors)
% sigma_min = floor on gamma_d (prevents data type domination)
% sigma_max = ceiling on gamma_d (generous, absorbs modeling error)
% skip_dtype = true if data type has no valid observations
[sigma_prior, sigma_min, sigma_max, skip_dtype] = prior_sigma_from_data(data);
prior.noise.sigma_prior = sigma_prior;
prior.noise.sigma_min   = sigma_min;
prior.noise.sigma_max   = sigma_max;
prior.noise.skip_dtype  = skip_dtype;

% correlated noise toggle
% ON: adds correlation length L per data type (27 total params)
% OFF: diagonal covariance, no L (23 total params)
prior.noise.use_correlated = false;
prior.noise.corrlen_names = {'L_hvsr', 'L_ellip', 'L_cph', 'L_ugr'};
prior.noise.corrlen_lb    = [0.01; 0.01; 0.01; 0.01];
prior.noise.corrlen_ub    = [1.50; 1.50; 1.50; 1.50];
prior.noise.corrlen_step  = [0.05; 0.05; 0.05; 0.05];
prior.noise.corrlen_init  = [0.10; 0.10; 0.10; 0.10];

% proposal step sizes for theta perturbations
prior.step_size = [ ...
    0.003; 0.010; 0.030; 0.15; 0.40; 0.50; 0.60; 0.80; 0.80; ...
    0.02;  0.03;  0.05;  0.06; 0.06; 0.05; 0.05; 0.05; 0.05; 0.05  ...
];

% MCMC settings
prior.mcmc.n_iterations = 24000;
prior.mcmc.burn_in      = 10000;
prior.mcmc.thin         = 25;
prior.mcmc.n_chains     = 10;
prior.mcmc.base_seed    = 12345;

% temperature annealing (Kirkpatrick et al. 1983, adapted from Eilon et al. 2018)
% tau premultiplies step sizes and tempers trial likelihood during burn-in
% tau = 1 + (tau_max - 1) * erfc(iter / (tau_cooldown / 3))
prior.mcmc.tau_max      = 4.0;
prior.mcmc.tau_cooldown = prior.mcmc.burn_in;

% paths
prior.paths.base       = '/expanse/lustre/projects/syu127/brotimi/HVSR-Joint-Inversion';
prior.paths.cps_bin    = '/expanse/lustre/projects/syu127/brotimi/Inversion/bin';
prior.paths.hv_dfa_bin = '/expanse/lustre/projects/syu127/brotimi/Inversion/bin';

end
