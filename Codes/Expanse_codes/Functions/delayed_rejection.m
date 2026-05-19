function [theta_out, log_L_out, pred_out, Phi_out, N_d_out, accepted, dr_stage] = ...
    delayed_rejection(theta, log_L, sigma_e, data, prior, corrlen, chol_cache, j, tau)
% Delayed rejection for theta proposals. Tries up to two perturbations
% before giving up. Stage 2 targets a neighbor of the first perturbed parameter.
% Tau scales step sizes via c5 and tempers the trial likelihood (Josh asymmetric approach).

n_params = prior.n_params;
idx_h = prior.idx.h;
idx_vs = prior.idx.vs;

% default outputs: keep our current state unchanged
theta_out = theta;
log_L_out = log_L;
pred_out  = struct('hvsr', [], 'ellip', [], 'cph', [], 'ugr', []);
Phi_out   = NaN(prior.noise.n_noise, 1);
N_d_out   = zeros(prior.noise.n_noise, 1);
accepted  = false;
dr_stage  = 0;

% stage 1: perturb parameter j from current accepted model
[theta_prop1, ~, is_feasible1, ~] = c5_perturb_theta(theta, prior, j, tau);

if is_feasible1
    [log_L_prop1, ~, pred_prop1, Phi_prop1, N_d_prop1, ~] = ...
        c3_compute_likelihood(theta_prop1, sigma_e, data, prior, corrlen, chol_cache);

    if isfinite(log_L_prop1)
        % asymmetric tempering: log(tau) added to trial likelihood only
        log_alpha1 = (log_L_prop1 - log_L) + log(tau);
        if log(rand()) < log_alpha1
            theta_out = theta_prop1;
            log_L_out = log_L_prop1;
            pred_out  = pred_prop1;
            Phi_out   = Phi_prop1;
            N_d_out   = N_d_prop1;
            accepted  = true;
            dr_stage  = 1;
            return;
        end
    end
end

% stage 2: first perturbation was rejected, target a neighbor parameter
if is_feasible1
    base_theta = theta_prop1;
else
    base_theta = theta;
end

% select neighbor of j based on physical coupling
j2 = pick_neighbor(j, idx_h, idx_vs, n_params);

[theta_prop2, ~, is_feasible2, ~] = c5_perturb_theta(base_theta, prior, j2, tau);

if ~is_feasible2
    return;
end

% evaluate doubly-perturbed model
[log_L_prop2, ~, pred_prop2, Phi_prop2, N_d_prop2, ~] = ...
    c3_compute_likelihood(theta_prop2, sigma_e, data, prior, corrlen, chol_cache);

if ~isfinite(log_L_prop2)
    return;
end

% compare against the original accepted state with asymmetric tempering
log_alpha2 = (log_L_prop2 - log_L) + log(tau);
if log(rand()) < log_alpha2
    theta_out = theta_prop2;
    log_L_out = log_L_prop2;
    pred_out  = pred_prop2;
    Phi_out   = Phi_prop2;
    N_d_out   = N_d_prop2;
    accepted  = true;
    dr_stage  = 2;
end

end


function j2 = pick_neighbor(j, idx_h, idx_vs, n_params)
% Select a physically coupled neighbor parameter for delayed rejection
% h_k couples to vs_k and vs_{k+1} (velocities above and below the interface)
% vs_k couples to h_{k-1} and h_k (interfaces above and below the layer)

candidates = [];

loc_h = find(idx_h == j, 1);
if ~isempty(loc_h)
    % j is a thickness parameter h_k, couple to vs_k and vs_{k+1}
    candidates = [idx_vs(loc_h), idx_vs(min(loc_h+1, length(idx_vs)))];
end

loc_vs = find(idx_vs == j, 1);
if ~isempty(loc_vs)
    % j is a velocity parameter vs_k, couple to h_{k-1} and h_k
    if loc_vs > 1
        candidates = [candidates, idx_h(loc_vs - 1)];
    end
    if loc_vs <= length(idx_h)
        candidates = [candidates, idx_h(loc_vs)];
    end
end

% remove duplicates and self
candidates = unique(candidates);
candidates(candidates == j) = [];
candidates(candidates < 1 | candidates > n_params) = [];

% pick one randomly from candidates, fall back to random if none
if ~isempty(candidates)
    j2 = candidates(randi(length(candidates)));
else
    j2 = randi(n_params);
end

end
