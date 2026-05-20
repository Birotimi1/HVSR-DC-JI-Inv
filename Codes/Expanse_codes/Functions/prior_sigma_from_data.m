function [sigma_prior, sigma_min, sigma_max, skip_dtype] = prior_sigma_from_data(data)
%
% SIGMA DEFINITIONS:
%   sigma_i   = per-point measurement uncertainty from data (fixed, never changes)
%   gamma_d   = hierarchical noise multiplier per data type (sampled during MCMC)
%               gamma_d = 1.0 means "trust the data errors as given"
%               gamma_d < 1.0 means data errors are overestimated, squeeze them
%               gamma_d > 1.0 means data errors are underestimated, expand them
%   sigma_eff = gamma_d * sigma_i = total effective uncertainty at each point
%
% In our code, gamma_d is stored as sigma_e(d).
%
% This function sets:
%   sigma_prior = prior center for gamma_d (set to 1.0 for all data types)
%   sigma_min   = hard floor on gamma_d (0.1 for all, prevents any data type
%                 from dominating the joint likelihood)
%   sigma_max   = ceiling on gamma_d (50.0 for all, generous upper bound)
%   skip_dtype  = logical flag, true if a data type has no valid observations

fields = {'hvsr', 'ellip', 'cph', 'ugr'};
n_noise = length(fields);

% gamma_d centered at 1.0: start by trusting measurement errors
sigma_prior = ones(n_noise, 1);

% hard floor: gamma_d cannot shrink below 0.1 (errors can't be squeezed more than 10x)
sigma_min = 0.1 * ones(n_noise, 1);

% ceiling: gamma_d can grow up to 50 (errors can expand up to 50x)
sigma_max = 50.0 * ones(n_noise, 1);

% check which data types have valid observations
skip_dtype = false(n_noise, 1);

for d = 1:n_noise
    fname = fields{d};

    % check data type exists and has uncertainty field
    if ~isfield(data, fname) || ~isfield(data.(fname), 'sig')
        fprintf('[prior_sigma] %s: no .sig field found, skipping data type\n', fname);
        skip_dtype(d) = true;
        continue;
    end

    sig = data.(fname).sig(:);
    sig = sig(sig > 0 & isfinite(sig));

    if isempty(sig)
        fprintf('[prior_sigma] %s: no valid uncertainties, skipping data type\n', fname);
        skip_dtype(d) = true;
        continue;
    end

    % report the data error statistics for reference
    fprintf('[prior_sigma] %s: geomean(sigma_i)=%.4f, min=%.4f, max=%.4f (%d points), gamma_d prior=%.1f, floor=%.1f\n', ...
        fname, exp(mean(log(sig))), min(sig), max(sig), length(sig), sigma_prior(d), sigma_min(d));
end

n_active = sum(~skip_dtype);
fprintf('[prior_sigma] %d of %d data types active\n', n_active, n_noise);
if n_active == 0
    error('prior_sigma_from_data: no data types have valid uncertainties');
end

end
