function model = c2_initialize_hbi_model(prior)
% Initialize a random feasible model for the MCMC chain
%
% sigma_e(d) = gamma_d = hierarchical noise multiplier, initialized at 1.0
% gamma_d = 1.0 means trust data errors as given

persistent paths_set cps_bin_set hvf_bin_set

cps_bin = prior.paths.cps_bin;
hvf_bin = prior.paths.hv_dfa_bin;
func_path = fullfile(prior.paths.base, 'AGU_models', 'functions');

need_reset = isempty(paths_set) || ...
             ~strcmp(cps_bin_set, cps_bin) || ...
             ~strcmp(hvf_bin_set, hvf_bin);

if need_reset
    new_path = [cps_bin, ':', hvf_bin, ':/usr/bin:/bin:/usr/sbin:/sbin'];
    setenv('PATH', new_path);
    if ~contains(path, func_path)
        addpath(func_path);
    end
    cps_bin_set = cps_bin;
    hvf_bin_set = hvf_bin;
    paths_set = true;
end

lb = prior.bounds.lower;
ub = prior.bounds.upper;
n_params = prior.n_params;
n_noise = prior.noise.n_noise;

% draw random feasible theta
max_attempts = 10000;
theta = zeros(n_params, 1);
initialized = false;

for attempt = 1:max_attempts
    for i = 1:n_params
        theta(i) = lb(i) + rand() * (ub(i) - lb(i));
    end
    [is_feasible, ~] = check_feasibility_internal(theta, prior);
    if is_feasible
        initialized = true;
        break;
    end
end

if ~initialized
    error('c2_initialize_model: Failed to find feasible theta after %d attempts', max_attempts);
end

model.theta = theta;
model.n_params = n_params;

% initialize gamma_d at 1.0 (trust data errors as starting point)
% the MCMC will adjust each gamma_d to find the right noise level
model.sigma_e = ones(n_noise, 1);

model.log_likelihood = -Inf;
model.accept_count = zeros(n_params, 1);
model.reject_count = zeros(n_params, 1);

fprintf('[c2] Initialized theta:\n');
fprintf('  h (km): ');
fprintf('%.4f ', theta(prior.idx.h));
fprintf('\n');
fprintf('  Vs (km/s): ');
fprintf('%.4f ', theta(prior.idx.vs));
fprintf('\n');
fprintf('  Total depth: %.2f km\n', sum(theta(prior.idx.h)));
fprintf('[c2] Initialized gamma_d (noise multiplier): ');
fprintf('%.2f ', model.sigma_e);
fprintf('\n');

end


function [is_feasible, violation] = check_feasibility_internal(theta, prior)
% local feasibility check for initialization

is_feasible = true;
violation = '';

if isempty(theta)
    is_feasible = false;
    violation = 'theta is empty';
    return;
end

if any(~isfinite(theta))
    is_feasible = false;
    violation = 'theta contains non-finite values';
    return;
end

lb = prior.bounds.lower;
ub = prior.bounds.upper;
if any(theta < lb) || any(theta > ub)
    is_feasible = false;
    violation = 'theta outside bounds';
    return;
end

h = theta(prior.idx.h);
if any(h <= 0)
    is_feasible = false;
    violation = 'non-positive thickness';
    return;
end

if prior.enforce_monotonicity
    vs = theta(prior.idx.vs);
    if any(diff(vs) < 0)
        is_feasible = false;
        violation = 'Vs not monotonic';
        return;
    end
end

end
