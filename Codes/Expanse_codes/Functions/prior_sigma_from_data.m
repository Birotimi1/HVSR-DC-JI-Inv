function [sigma_prior, sigma_min] = prior_sigma_from_data(data)
% Compute prior center and minimum for sigma from observed data uncertainties
% sigma_prior is the geometric mean of data.dtype.sig per data type
% sigma_min is a hard floor at 10% of sigma_prior

fields = {'hvsr', 'ellip', 'cph', 'ugr'};
n_noise = length(fields);
sigma_prior = NaN(n_noise, 1);
sigma_min   = NaN(n_noise, 1);

for d = 1:n_noise
    fname = fields{d};

    % check data type exists and has observations
    if ~isfield(data, fname) || ~isfield(data.(fname), 'sig')
        fprintf('[prior_sigma] %s: no .sig field found, using default\n', fname);
        continue;
    end

    sig = data.(fname).sig(:);

    % remove zeros and NaNs
    sig = sig(sig > 0 & isfinite(sig));
    if isempty(sig)
        fprintf('[prior_sigma] %s: no valid uncertainties, using default\n', fname);
        continue;
    end

    % geometric mean as prior center
    sigma_prior(d) = exp(mean(log(sig)));

    % hard floor at 10% of prior center
    sigma_min(d) = 0.1 * sigma_prior(d);

    fprintf('[prior_sigma] %s: sigma_prior = %.4f, sigma_min = %.4f (%d points)\n', ...
        fname, sigma_prior(d), sigma_min(d), length(sig));
end

% fill missing with defaults
for d = 1:n_noise
    if isnan(sigma_prior(d))
        sigma_prior(d) = 1.0;
        sigma_min(d) = 0.01;
        fprintf('[prior_sigma] %s: using default sigma_prior = %.2f\n', fields{d}, sigma_prior(d));
    end
end

end
