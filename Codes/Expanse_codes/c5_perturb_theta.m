function [theta_prop, param_idx, is_feasible, violation] = c5_perturb_theta(theta, prior, param_idx_input, tau)
% Perturb a single theta parameter by a Gaussian step scaled by tau

theta = theta(:);
n_params = prior.n_params;

% select parameter index
if nargin < 3 || isempty(param_idx_input)
    param_idx = randi(n_params);
else
    param_idx = param_idx_input;
end

if param_idx < 1 || param_idx > n_params
    error('c5_perturb_theta: param_idx=%d out of range [1,%d]', param_idx, n_params);
end
if ~isfield(prior, 'step_size') || isempty(prior.step_size)
    error('c5_perturb_theta: prior.step_size is missing/empty.');
end
if numel(prior.step_size) ~= n_params
    error('c5_perturb_theta: prior.step_size length (%d) must equal n_params (%d).', ...
        numel(prior.step_size), n_params);
end

% base step size scaled by temperature
delta = prior.step_size(param_idx) * tau;

if ~isfinite(delta) || delta <= 0
    error('c5_perturb_theta: invalid step size for param %d: %g', param_idx, delta);
end

% global bounds
lb = prior.bounds.lower(:);
ub = prior.bounds.upper(:);
if numel(lb) ~= n_params || numel(ub) ~= n_params
    error('c5_perturb_theta: bounds length mismatch with n_params.');
end
L = lb(param_idx);
U = ub(param_idx);

% tighten bounds for monotonicity on Vs parameters
if isfield(prior, 'enforce_monotonicity') && prior.enforce_monotonicity ...
        && isfield(prior, 'idx') && isfield(prior.idx, 'vs')
    vs_idx = prior.idx.vs(:);
    loc = find(vs_idx == param_idx, 1, 'first');
    if ~isempty(loc)
        if loc > 1
            L = max(L, theta(vs_idx(loc-1)));
        end
        if loc < numel(vs_idx)
            U = min(U, theta(vs_idx(loc+1)));
        end
    end
end

% check local bounds validity
if ~isfinite(L) || ~isfinite(U) || L > U
    theta_prop = theta;
    is_feasible = false;
    violation = sprintf('Invalid local bounds for param %d: L=%.6g, U=%.6g', param_idx, L, U);
    return;
end

% Gaussian perturbation with reflecting boundary
eta = randn() * delta;
x = theta(param_idx) + eta;
x_ref = reflect_to_interval(x, L, U);

theta_prop = theta;
theta_prop(param_idx) = x_ref;

% final feasibility check
[is_feasible, violation] = c4_check_feasibility_internal(theta_prop, prior);

end


function x_ref = reflect_to_interval(x, a, b)
% Reflect x into [a, b] to preserve detailed balance

if a == b
    x_ref = a;
    return;
end
w = b - a;
if w <= 0 || ~isfinite(w)
    x_ref = min(max(x, a), b);
    return;
end
t = mod(x - a, 2*w);
if t > w
    t = 2*w - t;
end
x_ref = a + t;
if x_ref < a
    x_ref = a;
elseif x_ref > b
    x_ref = b;
end

end
