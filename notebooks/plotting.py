import hypothesis
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import palettable
import torch
import warnings

from hypothesis.stat import confidence_level
from hypothesis.stat import highest_density_level
from hypothesis.stat import likelihood_ratio_test_statistic
from hypothesis.visualization.util import make_square
from matplotlib import rc
from matplotlib.colors import LogNorm
from ratio_estimation import RatioEstimator
from scipy.stats import chi2
from sklearn.neighbors import KernelDensity
from util import MarginalizedAgePrior
from util import Prior



matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amssymb}"



# Matplotlib settings
plt.rcParams.update({"font.size": 16})
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
plt.rcParams["text.usetex"] = True

# Priors
prior_1d = MarginalizedAgePrior()
prior_2d = Prior()

# Plotting defaults
default_resolution = 100
masses_xticks = [0, 10, 20, 30, 40, 50]
extent = [ # I know, this isn't very nice :(
    prior_2d.low[0].item(), prior_2d.high[0].item(),
    prior_2d.low[1].item(), prior_2d.high[1].item()]



@torch.no_grad()
def plot_stream(ax, phi, stream):
    ax.step(phi, stream.reshape(-1), lw=2, color="black")
    ax.set_ylim([0, 2])
    ax.minorticks_on()
    make_square(ax)


@torch.no_grad()
def plot_1d_pdf(ax, pdf, resolution=default_resolution):
    masses = np.linspace(prior_1d.low.item(), prior_1d.high.item(), resolution)
    ax.minorticks_on()
    ax.plot(masses, pdf, lw=2, color="black")
    ax.set_xlabel(r"$m_\textsc{wdm}$")
    ax.set_xticks(masses_xticks)
    ax.grid(True, which="both", alpha=.15)
    make_square(ax)


@torch.no_grad()
def plot_1d_profile_likelihood(ax, profile_likelihood):
    resolution = len(profile_likelihood)
    masses = np.linspace(prior_1d.low.item(), prior_1d.high.item(), resolution)
    ax.minorticks_on()
    ax.plot(masses, profile_likelihood, lw=2, color="black")
    ax.set_xlabel(r"$m_\textsc{wdm}$")
    ax.set_xticks(masses_xticks)
    ax.set_ylabel(r"Profile likelihood")
    ax.set_ylabel(r"$-2\mathbb{E}\left[\log r(x\vert\vartheta) - \log r(x\vert\hat{\vartheta})\right]$")
    ax.grid(True, which="both", alpha=.15)
    make_square(ax)


@torch.no_grad()
def plot_1d_lr(ax, profile_likelihood):
    plot_1d_profile_likelihood(ax, profile_likelihood)


@torch.no_grad()
def plot_1d_pdf_std(ax, pdf, std, resolution=default_resolution):
    masses = np.linspace(prior_1d.low.item(), prior_1d.high.item(), resolution)
    ax.fill_between(masses, pdf - std, pdf + std, color="black", alpha=.15)
    make_square(ax)


@torch.no_grad()
def plot_2d_pdf(ax, pdf, mass=None, age=None):
    ax.minorticks_on()
    resolution = len(pdf)
    cmap = palettable.cmocean.sequential.Ice_20.mpl_colormap
    cmap = palettable.scientific.sequential.Oslo_3.mpl_colormap
    im = ax.imshow(pdf.T + 1, norm=LogNorm(), alpha=.75, interpolation="bilinear", extent=extent, origin="lower", cmap=cmap)
    if mass is not None and age is not None:
        ax.scatter(mass, age, s=150, marker='*', c="#ff4747", alpha=1.0, zorder=10)
    ax.set_xlabel(r"$m_{\textsc{wdm}}$")
    ax.set_xticks([10, 20, 30, 40, 50])
    ax.set_ylabel("Stream age in Gyr")
    ax.grid(True, which="both", alpha=.15, zorder=0, color="white")
    make_square(ax)



@torch.no_grad()
def plot_2d_profile_likelihood(ax, profile_likelihood, mass=None, age=None):
    ax.minorticks_on()
    resolution = len(profile_likelihood)
    cmap = palettable.scientific.sequential.Oslo_3_r.mpl_colormap
    im = ax.imshow(profile_likelihood.T + 1, alpha=.75, interpolation="bilinear", extent=extent, origin="lower", cmap=cmap)
    if mass is not None and age is not None:
        ax.scatter(mass, age, s=150, marker='*', c="#ff4747", alpha=1.0, zorder=10)
    ax.set_xlabel(r"$m_{\textsc{wdm}}$")
    ax.set_xticks([10, 20, 30, 40, 50])
    ax.set_ylabel("Stream age in Gyr")
    ax.grid(True, which="both", alpha=.15, zorder=0, color="white")
    make_square(ax)


@torch.no_grad()
def plot_2d_lr(ax, profile_likelihood, mass=None, age=None):
    plot_2d_profile_likelihood(ax, profile_likelihood, mass, age)


@torch.no_grad()
def plot_2d_confidence_level(ax, profile_likelihood, level=0.95, clabel=None):
    z = chi2.isf(1 - level, df=2)
    resolution = len(profile_likelihood)
    masses = torch.linspace(prior_2d.low[0], prior_2d.high[0] - 0.01, resolution).view(-1, 1)
    ages = torch.linspace(prior_2d.low[1], prior_2d.high[1] - 0.01, resolution).view(-1, 1)
    grid_ages, grid_masses = torch.meshgrid(masses.view(-1), ages.view(-1))
    c = ax.contour(grid_ages.cpu().numpy(), grid_masses.cpu().numpy(), profile_likelihood, [z], colors="white")
    if clabel is not None:
        fmt = {}
        fmt[c.levels[0]] = clabel
        ax.clabel(c, c.levels, inline=True, fmt=fmt)


@torch.no_grad()
def plot_2d_confidence_levels(ax, profile_likelihood):
    plot_2d_confidence_level(ax, profile_likelihood, level=0.68, clabel="68\%")
    plot_2d_confidence_level(ax, profile_likelihood, level=0.95, clabel="95\%")
    plot_2d_confidence_level(ax, profile_likelihood, level=0.997, clabel="99.7\%")
    plot_2d_confidence_level(ax, profile_likelihood, level=0.99993, clabel="99.993\%")
    plot_2d_confidence_level(ax, profile_likelihood, level=0.9999994, clabel="99.99994\%")


@torch.no_grad()
def plot_1d_confidence_level(ax, profile_likelihood, level=0.95, color="black"):
    z = chi2.isf(1 - level, df=1)
    n = len(profile_likelihood)
    masses = np.linspace(prior_1d.low.item(), prior_1d.high.item(), n)
    # Utility functions
    def clean_sign(signs):
        current = signs[0]
        for index in range(len(signs)):
            if signs[index] == 0:
                signs[index] = -current
            current = signs[index]

        return signs
    def plot(masses, level, color="black"):
        masses = masses.tolist()
        #masses.sort()
        edge_picked = False
        while len(masses) > 0:
            m = masses[0]
            end = None
            if profile_likelihood[0] <= level and not edge_picked:
                edge_picked = True
                start = prior_1d.low.item()
                end = m
            else:
                start = m
            masses.pop(0)
            if profile_likelihood[-1] <= level and end is None:
                end = prior_1d.high.item() - 0.01
            if end is None:
                end = masses[0]
                masses.pop(0)
            ax.plot([start, end], [0., 0.], lw=5, color=color)
    indices = np.argwhere(np.diff(clean_sign(np.sign(z - profile_likelihood)), axis=0).reshape(-1)).flatten()
    m = masses[indices]
    plot(m, z, color=color)


@torch.no_grad()
def plot_1d_confidence_levels(ax, profile_likelihood):
    plot_1d_confidence_level(ax, profile_likelihood, level=0.997, color="#dddddd")
    plot_1d_confidence_level(ax, profile_likelihood, level=0.95, color="#aaaaaa")
    plot_1d_confidence_level(ax, profile_likelihood, level=0.68, color="black")


@torch.no_grad()
def plot_1d_contours(ax, pdf):
    n = len(pdf)
    masses = np.linspace(prior_1d.low.item(), prior_1d.high.item(), n)
    # Compute the levels of the highest density regions
    level_1sigma = highest_density_level(pdf, 0.68)
    level_2sigma = highest_density_level(pdf, 0.95)
    level_3sigma = highest_density_level(pdf, 0.997, min_epsilon=10e-19)
    # Utility functions
    def clean_sign(signs):
        current = signs[0]
        for index in range(len(signs)):
            if signs[index] == 0:
                signs[index] = -current
            current = signs[index]

        return signs
    def plot(masses, level, color="black"):
        masses = masses.tolist()
        #masses.sort()
        edge_picked = False
        while len(masses) > 0:
            m = masses[0]
            end = None
            if pdf[0] > level and not edge_picked:
                edge_picked = True
                start = prior_1d.low.item()
                end = m
            else:
                start = m
            masses.pop(0)
            if pdf[-1] > level and end is None:
                end = prior_1d.high.item() - 0.01
            if end is None:
                end = masses[0]
                masses.pop(0)
            ax.plot([start, end], [0., 0.], lw=5, color=color)
    # Compute the intersections
    indices = np.argwhere(np.diff(clean_sign(np.sign(level_3sigma - pdf)), axis=0).reshape(-1)).flatten()
    m_1 = masses[indices]
    plot(m_1, level_3sigma, color="#dddddd")
    indices = np.argwhere(np.diff(clean_sign(np.sign(level_2sigma - pdf)), axis=0).reshape(-1)).flatten()
    m_2 = masses[indices]
    plot(m_2, level_2sigma, color="#aaaaaa")
    indices = np.argwhere(np.diff(clean_sign(np.sign(level_1sigma - pdf)), axis=0).reshape(-1)).flatten()
    m_3 = masses[indices]
    plot(m_3, level_1sigma, color="black")

    return m_1, m_2, m_3


@torch.no_grad()
def plot_2d_contours(ax, pdf, resolution=default_resolution):
    clabels = [
        r"${99.99994}\%$",
        r"${99.993}\%$",
        r"${99.7}\%$",
        r"${95}\%$",
        r"${68}\%$"]
    level_1sigma = highest_density_level(pdf, 0.68)
    level_2sigma = highest_density_level(pdf, 0.95)
    level_3sigma = highest_density_level(pdf, 0.997)
    level_4sigma = highest_density_level(pdf, 0.99993)
    level_5sigma = highest_density_level(pdf, 0.9999994)
    levels = [
        level_5sigma,
        level_4sigma,
        level_3sigma,
        level_2sigma,
        level_1sigma]
    # Draw the levels
    masses = torch.linspace(prior_2d.low[0], prior_2d.high[0] - 0.01, resolution).view(-1, 1)
    ages = torch.linspace(prior_2d.low[1], prior_2d.high[1] - 0.01, resolution).view(-1, 1)
    grid_ages, grid_masses = torch.meshgrid(masses.view(-1), ages.view(-1))
    c = ax.contour(grid_ages.cpu().numpy(), grid_masses.cpu().numpy(), pdf, levels, colors="white")
    fmt = {}
    for l, s in zip(c.levels, clabels):
        fmt[l] = s
    ax.clabel(c, c.levels, inline=True, fmt=fmt)


@torch.no_grad()
def compute_1d_pdf_abc(mock_path, resolution=default_resolution, bandwidth=2.5):
    # Prepare the posterior samples
    inputs = np.linspace(prior_1d.low.item(), prior_1d.high.item(), resolution).reshape(-1, 1)
    suffix = mock_path.split('/')[-1].split('-')[1]
    samples = np.load("out/mock/abc-" + suffix + "/samples.npy")[:, 0] # Masses only
    samples = samples.reshape(-1, 1)
    # Fit a kernel
    kernel = KernelDensity(kernel="epanechnikov", bandwidth=bandwidth).fit(samples)
    pdf = np.exp(kernel.score_samples(inputs))

    return pdf


@torch.no_grad()
def compute_2d_pdf_abc(mock_path, resolution=default_resolution, bandwidth=2.5):
    masses = torch.linspace(prior_2d.low[0], prior_2d.high[0] - 0.01, resolution).view(-1, 1).cpu()
    ages = torch.linspace(prior_2d.low[1], prior_2d.high[1] - 0.01, resolution).view(-1, 1).cpu()
    grid_masses, grid_ages = torch.meshgrid(masses.view(-1), ages.view(-1))
    inputs = torch.cat([grid_masses.reshape(-1,1), grid_ages.reshape(-1, 1)], dim=1)
    inputs = inputs.view(-1, 2).numpy()
    # Prepare the posterior samples
    suffix = mock_path.split('/')[-1].split('-')[1]
    samples = np.load("out/mock/abc-" + suffix + "/samples.npy")
    samples = samples.reshape(-1, 2)
    # Fit a kernel
    kernel = KernelDensity(kernel="epanechnikov", bandwidth=bandwidth).fit(samples)
    pdf = np.exp(kernel.score_samples(inputs)).reshape(resolution, resolution)

    return pdf


@torch.no_grad()
def compute_1d_profile_likelihood(r, densities, resolution=default_resolution, median=False):
    inputs = torch.linspace(prior_1d.low, prior_1d.high - 0.01, resolution)
    inputs = inputs.to(hypothesis.accelerator).view(-1, 1)
    profile_likelihoods = []
    for density in densities:
        observables = density.view(1, -1).repeat(resolution, 1)
        observables = observables.to(hypothesis.accelerator)
        log_ratios = r.log_ratio(inputs=inputs, outputs=observables)
        test_statistic = likelihood_ratio_test_statistic(log_ratios).view(1, -1).cpu()
        profile_likelihoods.append(test_statistic)
    profile_likelihoods = torch.cat(profile_likelihoods, dim=0)
    if median:
        m = profile_likelihoods.median(dim=0).values.numpy()
    else:
        m = profile_likelihoods.mean(dim=0).numpy()
    s = profile_likelihoods.std(dim=0).numpy()

    return m, s


@torch.no_grad()
def compute_1d_lr(r, densities, resolution=default_resolution, median=False):
    return compute_1d_profile_likelihood(r, densities, resolution=default_resolution, median=median)


@torch.no_grad()
def compute_1d_pdf(r, densities, resolution=default_resolution, median=False):
    log_pdfs = []
    inputs = torch.linspace(prior_1d.low, prior_1d.high - 0.01, resolution)
    inputs = inputs.to(hypothesis.accelerator).view(-1, 1)
    log_prior_probabilities = prior_1d.log_prob(inputs)
    log_prior_probabilities = log_prior_probabilities.to(hypothesis.accelerator)
    for density in densities:
        observables = density.view(1, -1).repeat(resolution, 1)
        observables = observables.to(hypothesis.accelerator)
        log_ratios = r.log_ratio(inputs=inputs, outputs=observables)
        log_pdfs.append((log_prior_probabilities + log_ratios).view(1, -1).cpu())
    log_pdfs = torch.cat(log_pdfs, dim=0)
    pdf = log_pdfs.exp()
    if median:
        m = pdf.median(dim=0).values.numpy()
    else:
        m = pdf.mean(dim=0).numpy()
    s = pdf.std(dim=0).numpy()

    return m, s


@torch.no_grad()
def compute_2d_pdf(r, densities, resolution=default_resolution, median=False):
    log_pdfs = []
    masses = torch.linspace(prior_2d.low[0], prior_2d.high[0] - 0.01, resolution).view(-1, 1)
    masses = masses.to(hypothesis.accelerator)
    ages = torch.linspace(prior_2d.low[1], prior_2d.high[1] - 0.01, resolution).view(-1, 1)
    ages = ages.to(hypothesis.accelerator)
    grid_masses, grid_ages = torch.meshgrid(masses.view(-1), ages.view(-1))
    inputs = torch.cat([grid_masses.reshape(-1,1), grid_ages.reshape(-1, 1)], dim=1)
    inputs = inputs.to(hypothesis.accelerator).view(-1, 2)
    log_prior_probabilities = prior_2d.log_prob(inputs).sum(dim=1).view(-1, 1)
    log_prior_probabilities = log_prior_probabilities.to(hypothesis.accelerator)
    for density in densities:
        observables = density.view(1, -1).repeat(resolution ** 2, 1)
        observables = observables.to(hypothesis.accelerator)
        log_ratios = r.log_ratio(inputs=inputs, outputs=observables)
        log_posterior = (log_prior_probabilities + log_ratios).view(1, resolution, resolution).cpu()
        log_pdfs.append(log_posterior)
    log_pdfs = torch.cat(log_pdfs, dim=0)
    pdf = log_pdfs.exp()
    if median:
        m = pdf.median(dim=0).values.numpy()
    else:
        m = pdf.mean(dim=0).numpy()
    s = pdf.std(dim=0).numpy()

    return m, s


@torch.no_grad()
def compute_2d_profile_likelihood(r, densities, resolution=default_resolution, median=False):
    masses = torch.linspace(prior_2d.low[0], prior_2d.high[0] - 0.01, resolution).view(-1, 1)
    masses = masses.to(hypothesis.accelerator)
    ages = torch.linspace(prior_2d.low[1], prior_2d.high[1] - 0.01, resolution).view(-1, 1)
    ages = ages.to(hypothesis.accelerator)
    grid_masses, grid_ages = torch.meshgrid(masses.view(-1), ages.view(-1))
    inputs = torch.cat([grid_masses.reshape(-1,1), grid_ages.reshape(-1, 1)], dim=1)
    inputs = inputs.to(hypothesis.accelerator).view(-1, 2)
    profile_likelihoods = []
    for density in densities:
        observables = density.view(1, -1).repeat(resolution ** 2, 1)
        observables = observables.to(hypothesis.accelerator)
        log_ratios = r.log_ratio(inputs=inputs, outputs=observables)
        test_statistic = likelihood_ratio_test_statistic(log_ratios).view(1, resolution, resolution).cpu()
        profile_likelihoods.append(test_statistic)
    profile_likelihoods = torch.cat(profile_likelihoods, dim=0)
    if median:
        m = profile_likelihoods.median(dim=0).values.numpy()
    else:
        m = profile_likelihoods.mean(dim=0).numpy()
    s = profile_likelihoods.std(dim=0).numpy()

    return m, s

@torch.no_grad()
def compute_2d_lr(r, densities, resolution=default_resolution, median=False):
    return compute_2d_lr(r, densities, resolution=default_resolution, median=median)
