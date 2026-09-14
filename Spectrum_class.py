import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

np.random.seed(None)
seed = np.random.get_state()

class SimulatedSpectrum:
    def __init__(self, rest_wavelength, amplitude, sigma=1.0, continuum=1.0):
        """
        Initializes the simulated spectrum template.
        
        Parameters:
        - rest_wavelength (float): The central wavelength of the line (lambda_0).
        - amplitude (float): Height of emission line (+ value) or depth of absorption line (- value).
        - line_width (float): The standard deviation (sigma) of the Gaussian line profile.
        - continuum (float): The baseline level of background light.
        """
        self.rest_wavelength = rest_wavelength
        self.amplitude = amplitude
        self.sigma = sigma
        self.continuum = continuum
        

        # Speed of light in km/s constant
        self.c = 299792.458 

    def create_grids(self, wl_max, wl_min, num_points):
        # 2. Create a uniformly spaced LOG-wavelength grid
        ln_wl_grid = np.linspace(np.log(wl_min), np.log(wl_max), num_points)
        wave_grid = np.exp(ln_wl_grid) # This is your real wavelength array

        # 3. Derive the corresponding velocity grid from the log spacing
        # Each pixel step (d_ln_wl) corresponds to a constant velocity step (dv)
        d_ln_wl = ln_wl_grid[1] - ln_wl_grid[0]
        dv = d_ln_wl * self.c  # km/s per pixel

        # 4. Create a relative velocity grid centered at 0 km/s for your CCF
        vel_grid = (ln_wl_grid - ln_wl_grid[num_points // 2]) * self.c


        return vel_grid, ln_wl_grid, wave_grid


    def generate_flux(self, vel_grid, wave_grid, noise_sigma=0.0, lines=100, seed=None, noise_seed=None, correlated_spectra=False):
        """
        Generates a numpy array of flux values based on a provided wavelength grid.
        
        Parameters:
        - vel_grid (numpy array): The x-axis radial velocities to evaluate.
        - velocity_shift (float): Doppler shift to apply to the line in km/s. Currently not used.
        - noise_sigma (float): Standard deviation of Gaussian noise to add.
        """
        vel_flux = np.zeros(len(vel_grid))
        if lines > 0:
            # Important to use the same seed since this method is called upon various times
            np.random.set_state(seed)

            # Measure the resolution step size of your velocity grid (km/s per pixel)
            dv = vel_grid[1] - vel_grid[0]
            
            # Convert the pixel sigma into velocity units to keep line widths consistent
            sigma_vel = self.sigma * dv

            # 2d-matrix method:

            # Create rng arrays for the amplitude and subpixel shift of our peaks 
            heights = np.abs(np.random.normal(0, self.amplitude, size=lines))
            if correlated_spectra:
                line_center_vel = np.random.uniform(vel_grid[0], vel_grid[-1], int(lines/2))
                shifts = np.random.uniform(-10*sigma_vel, 10*sigma_vel, int(lines/2)) + line_center_vel
                line_center_vel_2D = np.vstack((line_center_vel, shifts))
                line_center_vel = line_center_vel_2D.flatten()
            else:
                line_center_vel = np.random.uniform(vel_grid[0], vel_grid[-1], lines)

            # reshape the grid into a 2d array
            vel_grid_2d = vel_grid[:, np.newaxis]

            # Do the gaussain operation for the spectral line profile
            gaussian_matrix = heights * np.exp(-((vel_grid_2d - line_center_vel) ** 2) / (2 * sigma_vel ** 2))

            # Collapse the 2d array back into our 1-dimesional opacity array 
            tau = np.sum(gaussian_matrix, axis=1)
            
            # To garuantee that the flux is between 1 and 0
            vel_flux =  np.exp(-tau)
        if noise_seed == None:
            np.random.seed(None)
            noise_seed = np.random.get_state()
        np.random.set_state(noise_seed)
        # Adds random experimental noise if requested
        if noise_sigma > 0.0:
            noise = np.random.normal(0, noise_sigma, size=vel_grid.shape)
            vel_flux += noise
            
        current_wl_coords = self.rest_wavelength * (1.0 + vel_grid / self.c)

        interp_flux_func = interp1d(current_wl_coords, vel_flux, kind='cubic',
                                    bounds_error=False, fill_value=1.0)

        flux_wave = interp_flux_func(wave_grid)

        return vel_flux, flux_wave

lambda_0 = 545.1 # nm
c = 299792.458 # km/s

# Creation of spectra
gen_spec1 = SimulatedSpectrum(rest_wavelength=lambda_0, amplitude=0.5, sigma=2)
vel_axis, ln_wave_axis, wave_axis = gen_spec1.create_grids(560, 530, 10000)

# observed to template
template_vel_flux, template_wave_flux = gen_spec1.generate_flux(vel_axis, wave_axis, noise_sigma=0.0, lines=100, seed=seed, noise_seed=None)

# white noise same seed different sigma
seed_noises= []
wnoise_sigmas = [0.00, 0.1, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 3.00, 4.00, 5.00]
for i in wnoise_sigmas:
    wnoise_vel, wnoise_wave = gen_spec1.generate_flux(vel_axis, wave_axis, noise_sigma=i, lines=0, noise_seed=seed)
    seed_noises.append(wnoise_vel)


# added noise to simulate different observed data
obs_vel = []
for i in seed_noises:
    obs_vel.append(i + template_vel_flux)

# noise different seed same sigma
noises = []
for i in range(0, 12, 1):
    noise_vel, noise_wave = gen_spec1.generate_flux(vel_axis, wave_axis, noise_sigma=0.5, lines=0, noise_seed=None)
    noises.append(noise_vel)


# Auto Correlation
np.random.seed(None)
seed = np.random.get_state()

templates = []
line_list = [10, 50, 100, 250, 500, 1000]
noise_vel, noise_wave = gen_spec1.generate_flux(vel_axis, wave_axis, noise_sigma=0.5, lines=0, seed=seed)
for i in line_list:
    templates.append(gen_spec1.generate_flux(vel_axis, wave_axis, noise_sigma=0, lines=i, seed=seed)[0])

line_obs_vel = np.array(templates) + noise_vel
line_obs_vel = [row for row in line_obs_vel]

# correlated Spectra
correlated_templates = []
for i in line_list:
    correlated_templates.append(gen_spec1.generate_flux(vel_axis, wave_axis, noise_sigma=0, lines=i, seed=seed, correlated_spectra=True)[0])



# Plotting

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5))

# Plot the flux against the main BOTTOM axis (Velocity)
ax1.plot(vel_axis, template_vel_flux, color='black', lw=1.5, label='Simulated Spectrum')
ax1.set_xlabel('Relative Radial Velocity ($v$ in km/s)', fontsize=11)
ax1.set_ylabel('Flux (A.U.)', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.4)

# Create the secondary TOP axis linked to the bottom axis using our math functions
ax2.plot(wave_axis, template_wave_flux, color='red', lw=1.5)
ax2.set_xlabel('Physical Wavelength (λ in nm)', fontsize=11, labelpad=10)
ax2.set_ylabel('Flux (A.U.)', fontsize=11)

fig.suptitle('Simulated Spectra with ~100 lines')
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(11, 5))

axes = axes.flatten()

for i in range(1, 5, 1):
    ax = axes[i-1]

    ax.scatter(wave_axis, obs_vel[i], alpha=0.7, color='grey', s=2)
    ax.plot(wave_axis, template_wave_flux, lw=1.5, color='b')
    ax.set_xlabel('Physical Wavelength (λ in nm)')
    ax.set_ylabel('Flux (A.U.)')
    ax.set_title(f'noise \u03C3 = {wnoise_sigmas[i]}')
    ax.grid(True, linestyle='--', alpha=0.4)


fig.suptitle('Same spectra with added noise of different strenghts')
plt.tight_layout()
plt.show()


# plot correlated spectra
plt.plot(vel_axis, correlated_templates[0], color='black', lw=1.5, label='Simulated Spectrum')
plt.xlabel('Relative Radial Velocity ($v$ in km/s)', fontsize=11)
plt.ylabel('Flux (A.U.)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()




def compute_ccf_at_velocity(vel_grid, noisy_flux, template_flux, shift_v):
    """
    Computes a single Pearson correlation coefficient score 
    for a continuous, non-integer velocity shift.
    """
    # 1. Transform absorption dips back into positive features for clean matching
    data_features = 1.0 - noisy_flux
    template_features = 1.0 - template_flux
    
    # 2. Shift the template coordinates continuously in velocity space
    # New coordinates = original velocity positions + your continuous shift target
    shifted_vel_coords = vel_grid + shift_v
    
    # 3. Use a cubic spline to interpolate the shifted features back onto the regular grid
    interp_func = interp1d(shifted_vel_coords, template_features, kind='cubic', 
                           bounds_error=False, fill_value=0.0)
    shifted_template = interp_func(vel_grid)
    
    # 4. Return the scalar correlation coefficient (the standard sum of products / normalization)
    correlation_matrix = np.corrcoef(data_features, shifted_template)
    # ccf = np.sum(shifted_template*data_features) / (np.sqrt(np.var(shifted_template)*np.var(data_features))*len(vel_grid))

    return correlation_matrix[0, 1]


# Define a clean, high-resolution velocity grid to evaluate the full shape of the CCF
rv_evaluation_grid = np.linspace(-200.0, 200.0, 1000)


# Autocorrelation
idx = 0
auto_correlations = []
for i in templates:
    ccf_profile = [compute_ccf_at_velocity(vel_axis, i, i, v) for v in rv_evaluation_grid]
    auto_correlations.append(ccf_profile)

fig, axes = plt.subplots(3, 2, figsize=(11, 5))

axes = axes.flatten()
idx = 0
for i in auto_correlations:
    ax = axes[idx]
    
    ax.plot(rv_evaluation_grid, i, color='crimson', lw=2)
    ax.set_xlabel('radial velocity (km/s)')
    ax.set_title(f'{line_list[idx]} lines')
    ax.grid(True, linestyle='--', alpha=0.4)
    idx += 1

fig.suptitle('Auto Correlation', x=0.53, size=25)
fig.supylabel('pearson coefficient')
plt.tight_layout()
plt.show()


# Auto-correlation for correlated spectra
idx = 0
auto_correlations = []
for i in correlated_templates:
    ccf_profile = [compute_ccf_at_velocity(vel_axis, i, i, v) for v in rv_evaluation_grid]
    auto_correlations.append(ccf_profile)

fig, axes = plt.subplots(3, 2, figsize=(11, 5))

axes = axes.flatten()
idx = 0
for i in auto_correlations:
    ax = axes[idx]
    
    ax.plot(rv_evaluation_grid, i, color='crimson', lw=2)
    ax.set_xlabel('radial velocity (km/s)')
    ax.set_title(f'{line_list[idx]} lines')
    ax.grid(True, linestyle='--', alpha=0.4)
    idx += 1

fig.suptitle('Auto Correlation', x=0.53, size=25)
fig.supylabel('pearson coefficient')
plt.tight_layout()
plt.show()


correlations = []
for i, j in zip(line_obs_vel, templates):
    ccf_profile = [compute_ccf_at_velocity(vel_axis, i, j, v) for v in rv_evaluation_grid]
    correlations.append(ccf_profile)

fig, axes = plt.subplots(3, 2, figsize=(11, 5))

axes = axes.flatten()
idx = 0
for i in correlations:
    ax = axes[idx]
    
    ax.plot(rv_evaluation_grid, i, color='crimson', lw=2)
    ax.set_xlabel('radial velocity (km/s)')
    ax.set_title(f'{line_list[idx]} lines')
    ax.grid(True, linestyle='--', alpha=0.4)
    idx += 1

fig.suptitle('Cross-Correlation with noisy signal, (noise \u03C3 = 0.5)', x=0.54, size=18)
fig.supylabel('pearson coefficient')
plt.tight_layout()
plt.show()


# Smaller evaluation grid for faster
rv_evaluation_grid = np.linspace(-100.0, 100.0, 500)


noise_coefs = []
idx = 0
legend = []
for i in seed_noises:
    coef = compute_ccf_at_velocity(vel_axis, i, template_vel_flux, 0)
    noise_coefs.append(coef)
    legend.append(f"\u03C3 {wnoise_sigmas[idx]}")
    idx += 1

print(noise_coefs)
plt.plot(wnoise_sigmas, noise_coefs)
plt.xlabel("Scaling strenght of white noise")
plt.ylabel("Pearson Coefficient")
plt.title("Correlation strength as a function noise strenght for same noise seed")
plt.show()


#
idx = 0
for i in obs_vel:
    ccf_wnoise_profile = [compute_ccf_at_velocity(vel_axis, i, template_vel_flux, v) for v in rv_evaluation_grid]
    plt.plot(rv_evaluation_grid, ccf_wnoise_profile, lw=2.0)
    idx += 1
plt.title("Cross-correlation of template and a noisy template for each noise strenght")
plt.plot([0, 7], [np.min(noise_coefs), np.min(noise_coefs)])
plt.legend(legend)
plt.ylabel('Pearson Coefficient')
plt.xlabel('radial velcoity (km/s)')
plt.show()



# a bunch of noise
noise_lst = []
for i in range(1, 101):
    np.random.seed(i)
    noise = np.random.normal(0, 0.5, 10000)
    noise_lst.append(noise)




# noise with no signal 
noise_coefs = []
for i in noise_lst:
    coef = compute_ccf_at_velocity(vel_axis, i, template_vel_flux, 0)
    noise_coefs.append(coef)

mean_correlation_floor = np.mean(noise_coefs)

plt.scatter([i for i,_ in enumerate(noise_coefs)], noise_coefs)
plt.plot([0, 100], [mean_correlation_floor, mean_correlation_floor])
plt.xlabel("Different noise seeds")
plt.ylabel("Pearson coefficient")
plt.title("The spread of the noise floor coefficient")
plt.show()



lines = np.linspace(0, 3000, 120, dtype=int)
lines[0] = int(10)
lines_coefs = []

for i in lines:
    flux = gen_spec1.generate_flux(vel_axis, wave_axis, noise_sigma=0, lines=i, seed=seed)[0]
    noise = flux + noises[0]
    coef = compute_ccf_at_velocity(vel_axis, noise, flux, 0)
    lines_coefs.append(coef)
    if i == 3000:
        print(np.var(flux))



plt.scatter(lines, lines_coefs, s=2)
plt.xlabel("# Spectral Lines")
plt.ylabel("Pearson Coefficient")
plt.title("Correlation strength as a function of spectral lines, (noise sigma = 0.5)")
plt.show()

# noise floor
noise_floor = compute_ccf_at_velocity(vel_axis, seed_noises[3], template_vel_flux, 0)

# Correlation coeeficient against noise sigma
coefs = []
for i in obs_vel:
    coef = np.corrcoef(i, template_vel_flux)
    coefs.append(coef[1, 0])

# SNRs for N(0, 1) stochastic variable
sigma_SNRs = []
for i in obs_vel:
    snr = np.var(template_vel_flux)
    sigma_SNRs.append(snr)

def theoretical_correlation_to_noise(noise_strenghts, snrs, correlation_floor):
    pearson_coefs = []
    cf = correlation_floor
    sqrt_SNR = np.sqrt(snrs)
    for i, j in zip(noise_strenghts, sqrt_SNR):
        pearson_coef = (1 + i/j*cf)/(np.sqrt(1 + (i/j)**2 + 2*i/j*cf))
        pearson_coefs.append(pearson_coef)
    return pearson_coefs

theo_coefs = theoretical_correlation_to_noise(wnoise_sigmas, sigma_SNRs, noise_floor)



print(coefs, theo_coefs)
plt.plot(wnoise_sigmas, coefs)
plt.plot(wnoise_sigmas, theo_coefs)
plt.axhline(noise_floor, 0, 6)
plt.yscale('log')
plt.xlabel("Noise Strength")
plt.ylabel("Pearson Correlation")
plt.legend(["Simulated result", "Theoretical expectation", "Noise Floor"])
plt.title("Signifigance as a function of noise strenght for noisy template")
plt.show()




# Objectives:
# - More data points for the strenght to lines plot
# - More data points for the strenght to noise plot
# - Direct comparison with theoretical graph in the plot
# - Clean up graphs to look more presentable
# - More clear labling
# Red Noise


# transform
def red_noise_gen(vel_axis, noise_sigma, continuum, noise_seed=None):
    if noise_sigma == 0:
        return [], [], np.zeros_like(vel_axis)
    if noise_seed == None:
        np.random.seed(None)
        noise_seed = np.random.get_state()
    np.random.set_state(noise_seed)

    # length
    num_points = len(vel_axis)

    # resolution
    dv = np.abs(vel_axis[1] - vel_axis[0])

    # Generate the white noise
    white_noise = np.random.normal(continuum, noise_sigma, num_points)

    # Fourier Transform
    white_fft = np.fft.rfft(white_noise)

    # Scaling factor
    frequencies = np.fft.rfftfreq(num_points, d=dv)
    scaling = np.zeros_like(frequencies)
    scaling[1:] = 1.0 / frequencies[1:]
    scaling[0] = 0.0

    # red fft
    red_fft = white_fft * scaling

    # red noise, normalised.
    red_noise = np.fft.irfft(red_fft, n=num_points)
    red_noise /= np.std(red_noise)
    red_noise = red_noise * noise_sigma

    return frequencies, white_fft, red_noise




frequencies, raw_fft, red_noise = red_noise_gen(vel_axis, 0.05, 0)

print(np.mean(red_noise), np.std(red_noise))

print(red_noise, np.shape(vel_axis))





# --- Plotting the Results ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: Wavelength Domain (Simulated Spectrum Noise)
ax1.plot(vel_axis, red_noise, color='crimson', lw=1.2)
ax1.set_title('Simulated Red Noise in Velocity Domain')
ax1.set_xlabel('Radial Velocity (km/s)')
ax1.set_ylabel('Noise Amplitude')
ax1.grid(True, alpha=0.3)

# Plot 2: Wavenumber Domain (Power Spectrum verification)
# Power is proportional to amplitude squared
power_spectrum = np.abs(raw_fft[:len(frequencies)] * (1.0 / np.where(frequencies==0, 1, frequencies)))**2
ax2.loglog(frequencies[1:], power_spectrum[1:], color='purple', alpha=0.5, label='Noise Power')
# Reference 1/k^2 line
ax2.loglog(frequencies[1:], (1/frequencies[1:])**2 * (power_spectrum[1]/ (1/frequencies[1])**2), 
           color='black', linestyle='--', label='$1/f^2$ ideal slope')

ax2.set_title('Power Spectral Density (Frequency Domain)')
ax2.set_xlabel('Frequnecy (Hz)')
ax2.set_ylabel('Power')
ax2.legend()
ax2.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()

rnoise = red_noise + np.random.normal(0, 0.25, len(vel_axis))
obs_rnoise_vel = template_vel_flux + rnoise

plt.plot(vel_axis, template_vel_flux, lw=2, c="b", alpha=0.2)
plt.scatter(vel_axis, obs_rnoise_vel, alpha=0.7, color='grey', s=2)
plt.xlabel("Radial velocity (Km/s)")
plt.ylabel("Flux (A.U.)")
plt.show()


obs_rnoise = []
for i in seed_noises:
    obs_rnoise.append(i+template_vel_flux+red_noise)

print("1")


ccfs_rnoise = []
for i in obs_rnoise:
    ccf_rnoise = [compute_ccf_at_velocity(vel_axis, i, template_vel_flux, v) for v in rv_evaluation_grid]
    ccfs_rnoise.append(ccf_rnoise)


# Plot all curves onto one single figure canvas
for i in ccfs_rnoise:
    plt.plot(rv_evaluation_grid, i, alpha=0.5) # alpha adds nice transparency if lines overlap

# Show the combined plot once the loop finishes
plt.show()


np.random.seed(None)
seed = np.random.get_state()
red_list = []
for i in wnoise_sigmas:
    frequencies, raw_fft, red_noise = red_noise_gen(vel_axis, i, 0, seed)
    red_list.append(red_noise)

for i in red_list[1:]:
    i += np.random.normal(0, 0.25, len(i))

fig, axes = plt.subplots(3, 2, figsize=(11, 5), sharex=True)

axes = axes.flatten()

# 2. Use your loop to target each axis dynamically via its index
for i in range(0, 6, 1):
    # Select the current axis using axes[i]
    ax = axes[i]
    
    ax.scatter(vel_axis, red_list[2*i], color='crimson', s=2)
    ax.set_title(f"Red noise \u03C3 {wnoise_sigmas[2*i]}") # Dynamically changes title
    ax.grid(True, linestyle='--', alpha=0.4)

fig.supylabel('Flux A.U.')
fig.supxlabel("Radial velocity (km/s)")
fig.suptitle("Red noise + white noise (noise \u03C3 = 0.25)")
plt.tight_layout()
plt.show()

obs_rnoise_vel = np.array(red_list) + template_vel_flux
obs_rnoise_vel = [row for row in obs_rnoise_vel]


ccfs_rnoise = []
for i in obs_rnoise_vel:
    ccf_rnoise = [compute_ccf_at_velocity(vel_axis, i, template_vel_flux, v) for v in rv_evaluation_grid]
    ccfs_rnoise.append(ccf_rnoise)

legend = []
idx = 0
for i in ccfs_rnoise:
    plt.plot(rv_evaluation_grid, i, alpha=0.5) # alpha adds nice transparency if lines overlap
    legend.append(f"\u03C3 {wnoise_sigmas[idx]}")
    idx += 1

# Show the combined plot once the loop finishes
plt.legend(legend)
plt.ylabel("Pearson coefficient")
plt.xlabel("Radial Velocity (km/s)")
plt.title("Statistical signifigance of peak in ccf function")
plt.show()

# Plot trend at v = 0 km/s


coeffs = []
for i in obs_rnoise_vel:
    coeffs.append(compute_ccf_at_velocity(vel_axis, i, template_vel_flux, 0))

plt.plot(wnoise_sigmas, coeffs, alpha=0.5, c="r") # alpha adds nice transparency if lines overlap
plt.plot(wnoise_sigmas, coefs)
plt.axhline(noise_floor, 0, 6)
plt.yscale("log")
plt.xlabel("noise strenght")
plt.ylabel("Pearson coefficient")
plt.legend(["Red noise", "White noise", "white noise floor"])
plt.show()

templates = []
line_list = [10, 50, 100, 250, 500, 1000]
for i in line_list:
    templates.append(gen_spec1.generate_flux(vel_axis, wave_axis, noise_sigma=0, lines=i, seed=seed)[0])

red_line = np.array(templates) + red_noise_gen(vel_axis, 0.5, 0)[2]
red_line = [row for row in red_line]

ccfs_rnoise = []
for obs, temp in zip(red_line, templates):
    ccf_rnoise = [compute_ccf_at_velocity(vel_axis, obs, temp, v) for v in rv_evaluation_grid]
    ccfs_rnoise.append(ccf_rnoise)


fig, axes = plt.subplots(3, 2, figsize = (15, 7))

axes = axes.flatten()

for i in range(0, 6, 1):

    ax = axes[i]
    
    ax.plot(rv_evaluation_grid, ccfs_rnoise[i], color='crimson', lw=2)
    ax.set_title(f"{line_list[i]} lines") 
    ax.grid(True, linestyle='--', alpha=0.4)

fig.supylabel('Flux A.U.')
fig.supxlabel('Relative Flux')
fig.suptitle("Cross-correlation of Red Noise for various spectral line denisties", size=16, x=0.53)
plt.tight_layout()
plt.show()
