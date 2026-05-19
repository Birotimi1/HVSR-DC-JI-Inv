function [sigma_prior, sigma_min, skip_dtype] = prior_sigma_from_data(data)
% Compute prior center and minimum for sigma from observed data uncertainties
% sigma_prior is the geometric mean of data.dtype.sig per data type
% sigma_min is a hard floor at 10% of sigma_prior
% skip_dtype is logical [4x1], true if data type should be skipped (no valid uncertainties)

fields = {'hvsr', 'ellip', 'cph', 'ugr'};
n_noise = length(fields);
sigma_prior = NaN(n_noise, 1);
sigma_min   = NaN(n_noise, 1);
skip_dtype  = false(n_noise, 1);

for d = 1:n_noise
    fname = fields{d};

    % check data type exists and has observations
    if ~isfield(data, fname) || ~isfield(data.(fname), 'sig')
        fprintf('[prior_sigma] %s: no .sig field found, skipping data type\n', fname);
        skip_dtype(d) = true;
        continue;
    end

    sig = data.(fname).sig(:);

    % remove zeros and NaNs
    sig = sig(sig > 0 & isfinite(sig));
    if isempty(sig)
        fprintf('[prior_sigma] %s: no valid uncertainties, skipping data type\n', fname);
        skip_dtype(d) = true;
        continue;
    end

    % geometric mean as prior center
    sigma_prior(d) = exp(mean(log(sig)));

    % hard floor at 10% of prior center
    sigma_min(d) = 0.1 * sigma_prior(d);

    fprintf('[prior_sigma] %s: sigma_prior = %.4f, sigma_min = %.4f (%d points)\n', ...
        fname, sigma_prior(d), sigma_min(d), length(sig));
end

% fill skipped data types with placeholder values
for d = 1:n_noise
    if skip_dtype(d)
        sigma_prior(d) = Inf;
        sigma_min(d) = Inf;
    end
end

n_active = sum(~skip_dtype);
fprintf('[prior_sigma] %d of %d data types active\n', n_active, n_noise);
if n_active == 0
    error('prior_sigma_from_data: no data types have valid uncertainties');
end

end
