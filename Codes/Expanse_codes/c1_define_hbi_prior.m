function prior = c1_define_hbi_prior_10layer(data)
% Define prior for 10-layer HBI inversion
% data is required to compute data-informed sigma priors

prior.n_params = 19;
prior.param_names = {'h1','h2','h3','h4','h5','h6','h7','h8','h9', ...
                     'Vs1','Vs2','Vs3','Vs4','Vs5','Vs6','Vs7','Vs8','Vs9','Vs10'};
prior.param_units = {'km','km','km','km','km','km','km','km','km', ...
                     'km/s','km/s','km/s','km/s','km/s','km/s','km/s','km/s','km/s','km/s'};

% parameter index mapping
prior.idx.h  = 1:9;
prior.idx.vs = 10:19;

% monotonicity constraint on Vs
prior.enforce_monotonicity = true;
prior.monotonic_start_idx  = 10;

% layer thickness bounds
h_min = [0.005; 0.02; 0.08; 0.30; 1.50; 2.00; 4.00; 6.00; 8.00];
h_max = [0.080; 0.30; 1.50; 4.00; 10.0; 15.0; 18.0; 22.0; 28.0];

% shear velocity bounds per layer
vs_min = [0.12; 0.25; 0.50; 0.80; 2.00; 2.60; 3.00; 3.40; 3.80; 4.10];
vs_max = [0.90; 1.60; 2.20; 3.20; 3.80; 4.00; 4.30; 4.60; 4.90; 5.20];

prior.bounds.lower = [h_min; vs_min];
prior.bounds.upper = [h_max; vs_max];
prior.depth_min_km = sum(h_min);
prior.depth_max_km = sum(h_max);

% noise parameters
prior.noise.n_noise = 4;
prior.noise.names   = {'sigma_hvsr','sigma_ellip','sigma_cph','sigma_ugr'};

% data-informed sigma prior from observed uncertainties from Zach meeting
[sigma_prior, sigma_min, skip_dtype] = prior_sigma_from_data(data);
prior.noise.sigma_prior = sigma_prior;
prior.noise.sigma_min   = sigma_min;
prior.noise.sigma_max   = sigma_prior * 50;
prior.noise.skip_dtype  = skip_dtype;

% correlated noise parameters
prior.noise.use_correlated = true;
prior.noise.corrlen_names = {'L_hvsr','L_ellip','L_cph','L_ugr'};
prior.noise.corrlen_lb    = [0.01; 0.01; 0.01; 0.01];
prior.noise.corrlen_ub    = [1.50; 1.50; 1.50; 1.50];
prior.noise.corrlen_step  = [0.05; 0.05; 0.05; 0.05];
prior.noise.corrlen_init  = [0.10; 0.10; 0.10; 0.10];

% proposal step sizes for theta
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

% temperature annealing
prior.mcmc.tau_max      = 4.0;
prior.mcmc.tau_cooldown = prior.mcmc.burn_in;

% paths
prior.paths.base       = '/expanse/lustre/projects/syu127/brotimi/HVSR-Joint-Inversion';
prior.paths.cps_bin    = '/expanse/lustre/projects/syu127/brotimi/Inversion/bin';
prior.paths.hv_dfa_bin = '/expanse/lustre/projects/syu127/brotimi/Inversion/bin';

end
