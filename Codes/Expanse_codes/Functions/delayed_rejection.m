function [theta_out, log_L_out, pred_out, Phi_out, N_d_out, accepted, dr_stage] = ...
    delayed_rejection(theta, log_L, sigma_e, data, prior, corrlen, chol_cache, j, tau)
% Delayed rejection for theta proposals. Tries up to two perturbations
% before giving up. Stage 2 applies a targeted compensating move.
% Tau scales step sizes via c5 and tempers the trial likelihood.
%
% Targeted perturbation options:
%   If h_k perturbed by +y (6 options):
%     a) h_(k+1) -= y
%     b) h_(k-1) -= y
%     c) h_(k+1) -= y AND vs_k += random
%     d) h_(k-1) -= y AND vs_(k-1) -= random
%     e) vs_k += random
%     f) vs_(k+1) += random
%   If vs_k perturbed (2 options):
%     g) h_(k-1) += random
%     h) h_k += random

n_params = prior.n_params;
idx_h = prior.idx.h;
idx_vs = prior.idx.vs;

% default outputs: keep current state unchanged
theta_out = theta;
log_L_out = log_L;
pred_out  = struct('hvsr', [], 'ellip', [], 'cph', [], 'ugr', []);
Phi_out   = NaN(prior.noise.n_noise, 1);
N_d_out   = zeros(prior.noise.n_noise, 1);
accepted  = false;
dr_stage  = 0;

% stage 1: perturb parameter j from current accepted model
[theta_prop1, ~, is_feasible1, ~] = c5_perturb_theta(theta, prior, j, tau);

% record perturbation magnitude for compensating moves in stage 2
delta_j = theta_prop1(j) - theta(j);

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

% stage 2: first perturbation rejected, apply targeted compensating move
if is_feasible1
    base_theta = theta_prop1;
else
    base_theta = theta;
    delta_j = 0;
end

% apply one of 6 (thickness) or 2 (velocity) targeted options
theta_prop2 = apply_targeted_perturbation(base_theta, j, delta_j, prior, tau);

% check feasibility of the compensated model
[is_feasible2, ~] = c4_check_feasibility_internal(theta_prop2, prior);
if ~is_feasible2
    return;
end

% evaluate the compensated model
[log_L_prop2, ~, pred_prop2, Phi_prop2, N_d_prop2, ~] = ...
    c3_compute_likelihood(theta_prop2, sigma_e, data, prior, corrlen, chol_cache);

if ~isfinite(log_L_prop2)
    return;
end

% compare against original accepted state with asymmetric tempering
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


function theta_out = apply_targeted_perturbation(theta, j, delta_j, prior, tau)
% Apply a targeted compensating perturbation based on what was perturbed
%
% For thickness h_k changed by +y (6 options):
%   a) h_(k+1) -= y
%   b) h_(k-1) -= y
%   c) h_(k+1) -= y AND vs_k += random
%   d) h_(k-1) -= y AND vs_(k-1) -= random
%   e) vs_k += random
%   f) vs_(k+1) += random
%
% For velocity vs_k changed (2 options):
%   g) h_(k-1) += random
%   h) h_k += random

theta_out = theta;
idx_h = prior.idx.h;
idx_vs = prior.idx.vs;
n_h = length(idx_h);
n_vs = length(idx_vs);

loc_h = find(idx_h == j, 1);
loc_vs = find(idx_vs == j, 1);

if ~isempty(loc_h)
    % j is thickness h_k, changed by delta_j
    options = {};

    % option a: h_(k+1) -= delta_j
    if loc_h < n_h
        opt.idx1 = idx_h(loc_h + 1);
        opt.val1 = -delta_j;
        opt.idx2 = 0;
        opt.val2 = 0;
        options{end+1} = opt;
    end

    % option b: h_(k-1) -= delta_j
    if loc_h > 1
        opt.idx1 = idx_h(loc_h - 1);
        opt.val1 = -delta_j;
        opt.idx2 = 0;
        opt.val2 = 0;
        options{end+1} = opt;
    end

    % option c: h_(k+1) -= delta_j AND vs_k += random
    if loc_h < n_h && loc_h <= n_vs
        opt.idx1 = idx_h(loc_h + 1);
        opt.val1 = -delta_j;
        opt.idx2 = idx_vs(loc_h);
        opt.val2 = tau * prior.step_size(idx_vs(loc_h)) * randn;
        options{end+1} = opt;
    end

    % option d: h_(k-1) -= delta_j AND vs_(k-1) -= random
    if loc_h > 1 && (loc_h - 1) <= n_vs
        opt.idx1 = idx_h(loc_h - 1);
        opt.val1 = -delta_j;
        opt.idx2 = idx_vs(loc_h - 1);
        opt.val2 = -abs(tau * prior.step_size(idx_vs(loc_h - 1)) * randn);
        options{end+1} = opt;
    end

    % option e: vs_k += random
    if loc_h <= n_vs
        opt.idx1 = idx_vs(loc_h);
        opt.val1 = tau * prior.step_size(idx_vs(loc_h)) * randn;
        opt.idx2 = 0;
        opt.val2 = 0;
        options{end+1} = opt;
    end

    % option f: vs_(k+1) += random
    if (loc_h + 1) <= n_vs
        opt.idx1 = idx_vs(loc_h + 1);
        opt.val1 = tau * prior.step_size(idx_vs(loc_h + 1)) * randn;
        opt.idx2 = 0;
        opt.val2 = 0;
        options{end+1} = opt;
    end

    % pick one option randomly and apply (uniform probability)
    if ~isempty(options)
        choice = options{randi(length(options))};
        theta_out(choice.idx1) = theta_out(choice.idx1) + choice.val1;
        if choice.idx2 > 0
            theta_out(choice.idx2) = theta_out(choice.idx2) + choice.val2;
        end
    end

elseif ~isempty(loc_vs)
    % j is velocity vs_k
    options = [];

    % option g: h_(k-1) += random
    if loc_vs > 1 && (loc_vs - 1) <= n_h
        options = [options, idx_h(loc_vs - 1)];
    end

    % option h: h_k += random
    if loc_vs <= n_h
        options = [options, idx_h(loc_vs)];
    end

    % pick one and apply random perturbation
    if ~isempty(options)
        j2 = options(randi(length(options)));
        theta_out(j2) = theta_out(j2) + tau * prior.step_size(j2) * randn;
    end
end

end
