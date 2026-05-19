function accept_info = make_accept_info(n_iter, n_saved, n_params, n_noise)
% Initialize tracking structure for MCMC diagnostics, tau history, and shake detection

% per-iteration tracking (full chain including burn-in)
accept_info.tau           = ones(n_iter, 1);
accept_info.log_L         = NaN(n_iter, 1);
accept_info.param_perturbed = zeros(n_iter, 1);
accept_info.accepted      = false(n_iter, 1);
accept_info.dr_stage      = zeros(n_iter, 1);
accept_info.is_shake      = false(n_iter, 1);

% per-saved-sample tracking (post burn-in only)
accept_info.sample_valid  = true(n_saved, 1);

% per-parameter acceptance tracking
accept_info.accept_count_theta  = zeros(n_params, 1);
accept_info.total_count_theta   = zeros(n_params, 1);
accept_info.accept_count_sigma  = zeros(n_noise, 1);
accept_info.total_count_sigma   = zeros(n_noise, 1);
accept_info.accept_count_corrlen = zeros(n_noise, 1);
accept_info.total_count_corrlen  = zeros(n_noise, 1);

% shake detection parameters
accept_info.shake_window        = 500;
accept_info.shake_std_fraction  = 0.1;
accept_info.shake_std_thresh    = NaN;
accept_info.shake_tau_max       = 3.0;
accept_info.shake_decay_rate    = 100;
accept_info.shake_tau_settled   = 1.01;
accept_info.shake_count         = 0;
accept_info.max_shakes          = 5;
accept_info.shake_iter_start    = 0;
accept_info.in_shake            = false;
accept_info.baseline_std        = NaN;

% rolling buffer for stagnation detection
accept_info.log_L_buffer     = NaN(accept_info.shake_window, 1);
accept_info.buffer_idx       = 0;

end
