import numpy as np
from astropy.utils.data import download_file  # or use curl

# To get info about the GW events
from pycbc.catalog import Catalog, Merger

# used for setting priors or initial points of walkers
from pycbc.distributions import JointDistribution, SinAngle, Uniform

# Signal processing and Matched filtering
from pycbc.filter import highpass, matched_filter, resample_to_delta_t, sigma
from pycbc.frame import read_frame  # read the GW data which is in gwf format

# likelihood and stochastic samplers
from pycbc.inference import models, sampler

# is inherited and used to make manual models
from pycbc.inference.models.base import BaseModel

# PSD estimation
from pycbc.psd import interpolate, inverse_spectrum_truncation

# td referes to time domain, there is equivalently frequency domain (get_fd_waveform)
from pycbc.waveform import get_td_waveform  # Generate GWaveforms
from scipy.stats import norm

# We need three things to do Bayesian inference 1) Prior 2) Generate waveform for given set of parameters,
# and calculate likelihood by weighted inner product: log likelihood ~ -(1/2) $\Sigma f(d -h, d-h)
# where d is the GW data in freq domain, h is the GW theoretical model for some parameter in freq domain,
# summation is over each detector and f(a,b) = integration((a*b) / psd) {weighted inner product}.
# SNR used in matched filtering can be derived from this expression using some simplified assumptions.
# This prior and likelihood gives us one point in the posterior sample space, and to map out full posterior we need
# 3) stochastic sampler (dynesty, emcee etc)


# 1D Gaussian
def gauss(x, mean, sigma):
    return np.exp((-((x - mean) ** 2)) / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)


# Information about catalogs, events
def info():
    m = Catalog(source="gwtc-3")  # Returns an object
    print("Events are : ")
    # All the events name
    for i in m:
        print(i)
    # Returns information about all events in Dictionary format, with keys as event names and values as object
    merger_events = m.mergers
    # Names of all the events in the given catalog
    print("Event names : ", merger_events.keys())  # Dictionary keys and values extraction
    # Returns just names # (Use above .keys method or this)
    merger_names = m.names
    # Returns dictionary with names and info website link about the event
    merger_details = m.data
    print(merger_details)
    # Returns parameter value for all events
    parameters = m.median1d("distance")
    # use any parameter Example mchirp = m.median1d('mchirp')

    # Compare functions of Merger and Catalog module both of which are under pycbc.catalog library
    d = Merger("GW170817")
    event_name = d.common_name
    # data about the event # Dictionary where keys are parameters that can be used in median1d like chirpmass, distance luminosity
    event_detail = d.data
    print("Details about the event : ", event_detail)
    frame = d.frame  # Not sure about this functionality????
    print("Frame of the event is ", frame)
    t = d.time  # time of the event
    print("TIme of the event occuring : ", t)
    par = d.median1d("distance")  # Returns parameter value
    redshift = d.median1d("redshift")
    # By default above returns parameters in the *source* frame.
    par = par * (1 + redshift)
    #  Converting it into detector frame by * (1 + z) coz all waveforms are detected by detector

    return event_name, event_detail, frame, t, par


# GW signal processing: 1. Passing through high pass filter 2. Whitening
def gw_data(detector, event_name):
    # Download the gravitational wave data for GW170817
    url = "https://dcc.ligo.org/public/0146/P1700349/001/{}-{}1_LOSC_CLN_4_V1-1187007040-2048.gwf"
    # Downloading "H-H1_LOSC_4_V2-1128678884-32.gwf" file or  use curl method to download
    fname = download_file(url.format(detector[0], detector[0]), cache=True)

    m = Merger(event_name)
    # read_frame(file_name, channel_name, start, end); channels names is typically IFO:LOSC-STRAIN, where IFO can be H1/L1/V1
    data = read_frame(fname, "{}:LOSC-STRAIN".format(detector), start_time=int(m.time - 260), end_time=int(m.time + 40))
    # Read the data directly from the Gravitational-Wave Frame (GWF) file. or
    # this data can be obtained by m = Merger("GW170817"); data = m.strain("H1")
    # This method by default gets the smallest version of the dataset. If additional data or specific versions are required,
    # use the method which is used in this.py file

    print("Sample rate of the downloaded file is ", data.sample_rate)

    # Signal processing

    # All we will see rn would be the low frequency behavior of the noise, since it is much louder
    # than the higher frequency noise and signal. Therefore, Remove low frequency bec noise of the instrument is at low freq
    data = highpass(data, 15.0)
    # Low pass filter removes high freq
    # However, after removing low freq there is clearly still some dominant frerquencies. To equalize this, we would need to apply a whitening filter.
    # We can also bandpass the data betweeen 30 - 250 Hz, by applying lowpass and highpass
    # simultaneously. This will remove frequency ranges which won't contribute

    # We will see two spikes around the end points which is because filter is applied in the frequnecy domain and the data is assumed
    # to be circular around the boundaries, but as the data has discontinues at end points, therefore we will see spikes.
    # To avoid this we trim the ends of the data sufficiently
    data = data.crop(2, 2)
    # Remove 2 seconds of data from both the beginning and end

    # This data is still colored gaussian and not whitened gaussian. We need to get flat and whitened gaussian noise.
    # Colored gaussian: different power at different frequencies.("Noise properties" different at different freq)
    # Whitening takes the data and attempts to make the power spectral density flat, so that all frequencies
    # contribute equally, and dominant frequencies can be removed.

    data = data.whiten(4, 4)
    # This produces a whitened set.
    # This works by estimating the power spectral density from the
    # data and then flattening the frequency response.
    # (1) The first option sets the duration in seconds of each
    #     sample of the data used as part of the PSD estimate.
    # (2) The second option sets the duration of the filter to apply

    #  We may also wish to resample the data if high frequency content is not important.
    # Resample data to 2048 Hz # Downsample to lower sampling rate # WHY???
    data = resample_to_delta_t(data, 1.0 / 2048)

    print("Downsampled to a lowering rate ", data.sample_rate)

    # Limit to times around the signal
    data = data.time_slice(m.time - 112, m.time + 16)
    # m.time = time of collision; m.start_time = start time of observation of data; m.end_time = end time of observation of data;
    # m.duration = duration of observation; m.sample_times = time array at the time of observation of each data point (used for plotting
    # timeseries strain) ; m.time_slice(start, end)

    # Convert to a frequency series by taking the data's FFT
    data_freq = data.to_frequencyseries()

    return data, data_freq


# Power spectral density of the time series data
def PSD(detector, event_name):
    data, data_freq = gw_data(detector, event_name)
    # Estimate the power spectral density of the data
    # This estimates the PSD by sub-dividing the data into 4s long segments. (See Welch's method)
    psd = data.psd(4)
    samplefreq = psd.sample_frequencies  # Similar to sample_times to plot psd

    # Now that we have the psd we need to interpolate it to match our data
    # and then limit the filter length of 1 / PSD. After this, we can
    # directly use this PSD to filter the data in a controlled manner
    psd = interpolate(psd, data.delta_f)

    # 1/PSD will now act as a filter with an effective length of 4 seconds
    # Since the data has been highpassed above 15 Hz, and will have low values
    # below this we need to informat the function to not include frequencies
    # below this frequency.
    psd = inverse_spectrum_truncation(psd, int(4 * psd.sample_rate), trunc_method="hann", low_frequency_cutoff=20.0)
    return psd


# Generate theoretical GWaveform and perform matched filtering
def gw_matchfilter(detector, event_name):
    data, data_freq = gw_data(detector, event_name)
    psd = PSD(detector, event_name)

    # The output of this function are the "plus" and "cross" polarizations of the gravitational-wave signal
    # as viewed from the line of sight at a given source inclination (assumed face-on if not provided)

    # In this case we "know" what the signal parameters are. In a search
    # we would grid over the parameters and calculate the SNR time series
    # for each one

    # We'll assume equal masses, which is within the posterior probability
    # of GW150914.
    m = 36  # Solar masses
    hp, hc = get_td_waveform(
        approximant="SEOBNRv4_opt",  # In this example, we've chosen to use the 'SEOBNRv4_opt' model. This approximant SEOBNRv4_opt models
        # inspiral, merger and ringdown. includes the ability for each black hole to spin in the same direction as the orbit (aligned spin).
        mass1=m,  # These theoretical waveforms need to be matched with waveforms generated by detector.
        # As detector generates waveforms in detector frame (obviously!), these theoretical waveforms too should have masses in detector frame.
        mass2=m,
        delta_t=1.0 / 4096,  # spacing between the time steps
        f_lower=30,  # lower freq of the range we consider of GW
    )  # Distance Luminostiy, Inclination and spin can also be added
    # generates hp = h+; hc = h* # what does detector detect? Combination of them?)
    # hp and hc maybe just shifted by a shift in simple systems of BBHs.
    # But if BBHs are precessing then it maybe complicated related to each other.
    # td referes to time domain, there is equivalently frequency domain (get_fd_waveform)

    # Notes:
    # Use hp.sample_times similar to Merger("").sample_times to plot theoretical waveforms
    # More the mass, smaller the signal and it is linearly related. 5M has twice the time duration of 10M.
    # More the distance, less the amplitude. Linear relationship
    # If you know what signal you are looking for in the data, then matched filtering is known to be the optimal method in Gaussian noise to
    # extract the siganl. Even when the parameters of the signal are unkown, one can test for each set of parameters one is interesting in
    # finding. Matched filtering is extracting optimal SNR.

    print(hp.shape)
    # We will resize the vector to match our data
    hp.resize(len(data))
    print(hp.shape)

    # We want our signal to be at the end of the data. Earlier in data we were having signal throughout, now we want the signal to be
    # at the end of the whole data, which is what is done by hp.resize(len(data)). This is done for technical reasons. This has some realtion
    # to SNR, but what????

    # The waveform begins at the start of the vector, so if we want the SNR time series to correspond to the approximate merger location
    # we need to shift the data so that the merger is approximately at the first bin of the data.

    # This function rotates the vector by a fixed amount of time. It treats the data as if it were on a ring. Note that
    # time stamps are *not* in general affected, but the true position in the vector is.

    # By convention waveforms returned from `get_td_waveform` have their merger stamped with time zero, so we can use the start time to
    # shift the merger into position
    template = hp.cyclic_time_shift(hp.start_time)

    # Matched filtering involves laying the potential signal over your data and integrating (after weighting frequencies correctly).
    # If there is a signal in the data that aligns with your 'template', you will get a large value when integrated over.
    snr = matched_filter(template, data, psd=psd, low_frequency_cutoff=20)

    # Remove time corrupted by the template filter and the psd filter
    # We remove 4 seonds at the beginning and end for the PSD filtering
    # And we remove 4 additional seconds at the beginning to account for
    # the template length (this is somewhat generous for
    # so short a template). A longer signal such as from a BNS, would
    # require much more padding at the beginning of the vector.
    snr = snr.crop(4 + 4, 4)

    peak = abs(snr).numpy().argmax()
    # Why am I taking an abs() here?
    # The `matched_filter` function actually returns a 'complex' SNR.
    # What that means is that the real portion correponds to the SNR
    # associated with directly filtering the template with the data.
    # The imaginary portion corresponds to filtering with a template that
    # is 90 degrees out of phase. Since the phase of a signal may be
    # anything, we choose to maximize over the phase of the signal.
    snrp = snr[peak]
    time = snr.sample_times[peak]
    # The time, amplitude, and phase of the SNR peak tell us how to align our proposed signal with the data.

    # Shift the template to the peak time
    dt = time - data.start_time
    aligned = template.cyclic_time_shift(dt)

    # scale the template so that it would have SNR 1 in this data
    aligned /= sigma(aligned, psd=psd, low_frequency_cutoff=20.0)

    # Scale the template amplitude and phase to the peak value
    aligned = (aligned.to_frequencyseries() * snrp).to_timeseries()
    aligned.start_time = data.start_time


# Prior; Priors are on parameters which will vary, use joint distribution
def prior_func(parameters):
    # Priors on the parameters which are varied
    inclination_prior = SinAngle(inclination=None)  # isotropic prior
    distance_prior = Uniform(distance=(10, 100))
    tc_prior = Uniform(tc=(m.time - 0.1, m.time + 0.1))
    prior = JointDistribution(parameters, inclination_prior, distance_prior, tc_prior)  # Joint prior
    # Gaussian(parameter_name=(.5, 2), mean=1, var=1) # mean 1 and var 1
    return prior


# All the Model (likelihood) names in the model module (Contains classes for calculating likelihoods)
def model_names():
    for i in models.models:
        print(i)


# test_eggbox test_normal test_rosenbrock test_volcano test_prior
# gaussian_noise marginalized_phase marginalized_polarization brute_parallel_gaussian_marginalize single_template relative

# Likelihoods

"""
The models starting with `test_` are analytic models. These have predefined likelihood functions
that are given by some standard distributions used in testing samplers. The other models are for GW astronomy:
they take in data and calculate a likelihood using an inner product between the data and a signal model.
Currently, all of the gravitational-wave models in PyCBC assume that the data is stationary Gaussian noise
in the absence of a signal. The difference between the models is they make varying simplfying assumptions,
in order to speed up likelihood evaluation. marginalized_phase, marginalized_polarization,
brute_parallel_gaussian_marginalize, single_template, relative assumes simplying assumptions.
"""


# Using TestNormal model: Analytical models (no data is used). Also it is possible to not provide prior to likelihood
# Analytic model is employed largely for testing the capabilities of different samplers.
def normal_likelihood(parameters, mean, prior):
    """
    parameters should be ((tuple of) string(s)) , mean (array-like, optional) Default=0,
    cov (array-like, optional) Default: diag terms =1 (var =1) , non diag =0.
    prior should be of class JOint distribution. All other parameters of likelihood should be in dictionary format
    """

    test_likelihood = models.TestNormal(parameters, mean=mean, prior=prior)
    return test_likelihood


#  Each model inherits from BaseModel. We are required to define a single method _likelihood
class ExampleModel_nodata(BaseModel):
    def __init__(self, param_name, **kwargs):  # priors can be added as kwargs as they are in dictionary format
        # Initialize the base model. It needs to know what the
        # variable parameters are, given here as (param_name).
        super(ExampleModel_nodata, self).__init__((param_name), **kwargs)  # Inherit init from BaseModel
        self.param = param_name

    def _loglikelihood(self):
        # self.current_params is a dictionary of the parameters
        # we need to evaluate the log(likelihood) for. The name
        # of the parameters are the ones we gave to the BaseModel init
        # method in the 'super' command above.
        pos = self.current_params[self.param]  # self.current_params is in Base Model

        # We'll use the logpdf of the normal distribution from scipy
        return norm.logpdf(pos)  # logpdf(x, loc=0, scale=1)


class ExampleModel_data(BaseModel):
    def __init__(self, data, **kwargs):  # priors can be added as kwargs as they are in dictionary format
        # We'll used fixed param names
        params = ("sigma", "mean")

        super(ExampleModel_data, self).__init__(params, **kwargs)
        self.data = data

    def _loglikelihood(self):
        # self.current_params is a dictionary of the parameters
        # we need to evaluate the log(likelihood) for.
        sigmasq = self.current_params["sigma"] ** 2.0
        mean = self.current_params["mean"]
        n = len(self.data)

        # log likihood for a normal distribution
        loglk = -n / 2.0 * np.log(2 * np.pi * sigmasq)
        loglk += -1.0 / (2 * sigmasq) * ((self.data - mean) ** 2.0).sum()
        return loglk


# data should be in frequency and dictionary format (keyed by observatory short name such as 'H1', 'L1', 'V1'),
#  static, low_freq_cutoff and psd should be in dict
def gw_likelihood(parameters, static, data_freq, psds, prior):

    """
    Using SingleTemplate model; contains the definition of the likelihood function you want to explore and
    details the parameters you are using. This model is useful when we know the intrinsic parameters of a source
    (i.e. component masses, spins), but we don't know the extrinsic parameters (i.e. sky location, distance, binary orientation)
    Fixing intrinsic parameters means that we don't have to generate waveforms at every single likelihood evaluation. BUt when
    estimating instrinsic p[arameter, we use MarginalizedPhaseGaussianNoise, which generates GWaveforms for each parameter it runs through.
    This model takes a new argument : waveform_transforms. Normally a merger signal is parameterized by its component masses, mass1 / mass2. We are going
    to estimate the chirp mass and try to sample in chirp mass and mass ratio. We can do this by providing the
    mapping between mchirp/q -> mass1/mass2. In this way we can use different parametizations than a model normally supports.
    """
    print("Likelihood calculation")
    lklhood = models.SingleTemplate(
        parameters,
        static_params=static,
        prior=prior,
        data_freq=data_freq,
        psds=psds,
        low_frequency_cutoff={"H1": 25, "L1": 25, "V1": 25},
        sample_rate=8192,
    )  # we can also add high_frequency_cutoff
    # Takes data of all the detectors, makes likelihood of all three and then multiply them to give us one likelihood func
    return lklhood


# Samplers
# Using Emcee sampler to run MCMC
def emcee(normal_likelihood, nwalkers, iterations, nprocesses):
    print("Number of nwalkers and iterations are ", nwalkers, iterations)
    engine = sampler.EmceeEnsembleSampler(normal_likelihood, nwalkers, nprocesses, use_mpi=True)
    # engine.set_p0(prior = Uniform(x = (-2,2), y = (-3,3))) # Set intial position of the walkers when prior is not given
    engine.set_p0()  # Initial position of walkers
    """
    Note that we do not need to provide anything to `set_p0` to set the initial positions
    if prior is there. By default, the sampler will draw from the prior. But when initial point needs to
    be given it should not be given as x =1, y =2 in the same way mean of likelihood can't be set as x = 0, y =1.
    example: models.TestNormal(('x', 'y'), mean = (x = 0, y = 1)) is wrong...just put (0,1) in mean. Here, in setting initial
    position of the walkers case, we need to put Uniform(x=(-1,1)).
    """
    print("Started running MCMC sampler")
    engine.run_mcmc(iterations)

    return engine


# While Emcee is sufficient for many problems, EmceePT, a parallel tempered version of Emcee is more effective at most GW data analysis problems.
def emceePT(gw_likelihood, temp, nwalkers, iterations, nprocesses):
    print("Number of nwalkers and iterations are ", nwalkers, iterations)
    # There is one additional parameter we need to give to EmcceePT which is the number of temperatures. The output of
    # this sampler will thus be 3-dimensional (temps x walkers x iterations). The 'coldest' temperature (0) will contain our actual results.
    engine = sampler.EmceePTSampler(gw_likelihood, ntemps=temp, nwalkers=nwalkers, nprocesses=nprocesses, use_mpi=True)
    # Number of temeratures to use in the sampler.
    engine.set_p0()  # If we don't set p0, it will use the models prior to draw initial points!
    print("MCMC run started")
    engine.run_mcmc(iterations)
    print("MCMC run done")

    return engine


if __name__ == "__main__":

    # Global variables
    # Parameters of GW likelihood function definitions
    # parameters which will vary
    parameters = ("distance", "inclination", "tc")
    # parameters which will remain constant
    static = {
        "mass1": 1.3757,  # Units in solar masses
        "mass2": 1.3757,
        "f_lower": 25.0,  # lower freq cut off
        "approximant": "TaylorF2",  # Approximant : only inspiral waveform model, and it is fast
        "polarization": 0,
        "ra": 3.44615914,  # Sky locations # Units in radians
        "dec": -0.40808407,  # Units in radians
    }
    event_name = "GW170817"
    m = Merger(event_name)
    ifos = ["H1", "V1", "L1"]  # List of observatories we'll analyze
    # Storing GW data (timeseries) of all detectors in dictionary format
    data = {}
    # Storing GW data (frequency domain)  of all detectors in dictionary format
    data_freq = {}
    psds = {}  # Storing power spectral density  of all detectors in dictionary format
    for i in ifos:
        print("Reading GW timeseries and freq domain data for detector ", i)
        data[i], data_freq[i] = gw_data(detector=i, event_name=event_name)
        print("Estimating PSD for detector ", i)
        psds[i] = PSD(data[i])

    print("Estimating prior")
    prior = prior_func(parameters)

    print("Estimating Likelihood")
    # my_model = ExampleModel_nodata("x") # Manual likelihood with no data
    # data = norm.rvs(size=10000)
    # my_model = ExampleModel_data(data) # Manual likelihood with no data
    lklhood = gw_likelihood(parameters=parameters, static=static, data_freq=data_freq, psds=psds, prior=prior)
    # data and psd should be in dictionaries for gw likelihood

    # Paramters of MCMC run
    nwalkers = 500  # No. of chains/walkers to be used
    iterations = 300  # No. of iterations to be used
    nprocesses = 4  # no. of processes to be used
    temp = 3  # No. of temperatures to be used

    engine = emceePT(lklhood, temp=temp, nwalkers=nwalkers, iterations=iterations, nprocesses=nprocesses)

    # Collecting parameter values at each iteration of the walker at each temperature.
    par_final = np.zeros((temp, nwalkers, iterations * len(parameters)))
    # iterations * length of parameters coz appending column wise
    for i in range(len(parameters)):
        if i == 0:
            # no. of walkers for each parameters which needs to be varied
            par_final = engine.samples[parameters[i]]
            print(par_final.shape, "when i=0")
        else:
            par = engine.samples[parameters[i]]
            par_final = np.append(par_final, par, axis=2)
    par_final = np.append(par_final, engine.model_stats["loglikelihood"], axis=2)
    # Likelihood Takes data of all the detectors, makes likelihood of all three and then multiply them to give us one likelihood func
    # appending likelihood at each iterations (i.e. no. of iteratiosn columns)
    print("Shape of the Final results are ", par_final.shape)

    # Header of the table
    myheader = "Values of different chains at each iterations. Shape is = nchains * (iterations * no. of parameters).\
         Number of slices equal to temperature. Iterations are arranged columns wise and in order 'distance', 'inclination','tc' "
    print("If the above shape is in 3D, then table will have each slice aranged in 2D.")
    print("Writing table")

    # Writing the table
    with open("chainvalues.txt", "w") as outfile:
        for slice_2d in par_final:
            np.savetxt(outfile, slice_2d, fmt="%16.12e", header=myheader, delimiter=",")
            outfile.write("# New slice\n")
