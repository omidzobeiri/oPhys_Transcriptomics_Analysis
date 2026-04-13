import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from scipy.stats import zscore
from sklearn.decomposition import PCA
# from tqdm import tqdm
import scipy.signal as signal
from typing import Iterable, Union, Optional
import h5py

def fillna_interp1d(timestamps, input_signal):

    hasnan = np.isnan(input_signal)
    f = interp1d(timestamps[~hasnan],input_signal[~hasnan],bounds_error=False,fill_value="extrapolate") #type: ignore
    return f(timestamps)


def percentile_standardization(X, lower_percentile = 10, upper_percentile = 90):
    lower_ind = np.round(X.shape[0]*lower_percentile/100).astype(int)
    higher_ind = np.round(X.shape[0]*upper_percentile/100).astype(int)
    X = np.divide(X-np.mean(np.sort(X,axis=0)[lower_ind:higher_ind],axis=0),\
        np.std(np.sort(X,axis=0)[lower_ind:higher_ind],axis=0)) 

    return X


def get_stimulus_timestamps(h5_file):
    h5 = h5py.File(h5_file, "r")

    data = h5['data'][:]
    sync_sig = data[:, -1]
    time_stamps = data[:, 0]

    meta = eval(h5['meta'][()])

    line_labels = meta['line_labels']
    sample_freq = meta['ni_daq']['sample_rate']

    # Finding rising and falling edges
    falling_edges = {}
    rising_edges = {}

    for bit, label in enumerate(line_labels):
        if not label:
            label = 'sig_' + str(bit)

        bit_array = np.bitwise_and(sync_sig, 2 ** bit).astype(bool).astype(np.uint8)
        bit_changes = np.ediff1d(bit_array, to_begin=bit_array[0])
        falling_edges[label] = time_stamps[np.where(bit_changes == 255)]/sample_freq
        rising_edges[label] = time_stamps[np.where(bit_changes == 1)]/sample_freq
    
    #Test and correct for photodiode transition errors (look for the last pulse-train pattern, 3 short pulses)

    ptd_rise_diff = np.ediff1d(rising_edges['stim_photodiode'])
    short = np.where(np.logical_and(ptd_rise_diff > 0.1, ptd_rise_diff < 0.3))[0]
    medium = np.where(np.logical_and(ptd_rise_diff > 0.5, ptd_rise_diff < 1.5))[0]

    # find the time differences betwen photodiode rising edges:
    # between 0.1 and 0.3 s -> short
    # between 0.5 and 1.5 s -> medium
    # find tha largest medium that there is a short comes just before or one before that
    # ptd_start = that medium + 1 otherwise ptd_start = 4 
    ptd_start = 3
    for i in medium:
        if set(range(i - 2, i)) <= set(short):
            ptd_start = i + 1

    # if ptd_start > 3:
    #     print('ptd_start: ' + str(ptd_start))
    #     print("Photodiode events before stimulus start.  Deleted.")


    if rising_edges['stim_photodiode'].max() <= falling_edges['vsync_stim'].max():
        # print('photodiode ends before stim_vsync already.')
        ptd_end = len(ptd_rise_diff)
    else:
        # print('truncating photodiode to end before stim_vsync.')
        ptd_end = np.where(rising_edges['stim_photodiode'] > falling_edges['vsync_stim'].max())[0][0] - 1
    # print('ptd_end: ' +str(ptd_end)+ ' max photodiode ' + str(rising_edges['stim_photodiode'].max())+' max stim '+ str(falling_edges['vsync_stim'].max()))
    ptd_errors = []
    error_frames = np.where(ptd_rise_diff[ptd_start:ptd_end] < 1.8)[0] + ptd_start
    if len(error_frames)>0:
        # print("Photodiode error detected. Number of frames:", len(error_frames))
        ptd_errors = (rising_edges['stim_photodiode'][error_frames])
        photodiode_rise = np.delete(rising_edges['stim_photodiode'], error_frames)
        ptd_end -= len(error_frames)
        ptd_rise_diff = np.ediff1d(photodiode_rise)

    # Estimate delay between stim and photodiode

    stim_on_photodiode_idx = 60 + 120 * np.arange(0, ptd_end - ptd_start, 1)
    stim_on_photodiode = falling_edges['vsync_stim'][stim_on_photodiode_idx]
    photodiode_on = rising_edges['stim_photodiode'][ptd_start + np.arange(0, ptd_end - ptd_start, 1)]
    delay_rise = photodiode_on - stim_on_photodiode
    delay_mu = np.mean(delay_rise[:-1])
    delay_sd = np.std(delay_rise[:-1])
    # print("monitor delay: " + str(delay_mu) + '+,-' + str(delay_sd))

    # adjust stimulus time to incorporate monitor delay
    stim_time = falling_edges['vsync_stim'] + delay_mu
    return stim_time



def calc_deriv(x, time):
    dx = np.diff(x, prepend=np.nan)
    dt = np.diff(time, prepend=np.nan)
    return dx / dt


def _angular_change(summed_voltage: np.ndarray,
                    vmax: Union[np.ndarray, float]) -> np.ndarray:
    """
    Compute the change in degrees in radians at each point from the
    summed voltage encoder data.

    Parameters
    ----------
    summed_voltage: 1d np.ndarray
        The "unwrapped" voltage signal from the encoder, cumulatively
        summed. See `_unwrap_voltage_signal`.
    vmax: 1d np.ndarray or float
        Either a constant float, or a 1d array (typically constant)
        of values. These values represent the theoretical max voltage
        value of the encoder. If an array, needs to be the same length
        as the summed_voltage array.
    Returns
    -------
    np.ndarray
        1d array of change in degrees in radians from each point
    """
    delta_theta = np.diff(summed_voltage, prepend=np.nan) / vmax * 2 * np.pi
    return delta_theta


def _shift(
        arr: Iterable,
        periods: int = 1,
        fill_value: float = np.nan) -> np.ndarray:
    """
    Shift index of an iterable (array-like) by desired number of
    periods with an optional fill value (default = NaN).

    Parameters
    ----------
    arr: Iterable (array-like)
        Iterable containing numeric data. If int, will be converted to
        float in returned object.
    periods: int (default=1)
        The number of elements to shift.
    fill_value: float (default=np.nan)
        The value to fill at the beginning of the shifted array
    Returns
    -------
    np.ndarray (1d)
        Copy of input object as a 1d array, shifted.
    """
    if periods <= 0:
        raise ValueError("Can only shift for periods > 0.")
    if fill_value is None:
        fill_value = np.nan
    if isinstance(fill_value, float):
        # Circumvent issue if int-like array with np.nan as fill
        shifted = np.roll(arr, periods).astype(float)
    else:
        shifted = np.roll(arr, periods)
    shifted[:periods] = fill_value
    return shifted


def deg_to_dist(angular_speed: np.ndarray) -> np.ndarray:
    """
    Takes the angular speed (radians/s) at each step in radians, and
    computes the linear speed in cm/s.

    Parameters
    ----------
    angular_speed: np.ndarray (1d)
        1d array of angular speed in radians/s
    Returns
    -------
    np.ndarray (1d)
        Linear speed in cm/s at each time point.
    """
    wheel_diameter = 6.5 * 2.54  # 6.5" wheel diameter, 2.54 = cm/in
    running_radius = 0.5 * (
        # assume the animal runs at 2/3 the distance from the wheel center
        2.0 * wheel_diameter / 3.0)
    running_speed_cm_per_sec = angular_speed * running_radius
    return running_speed_cm_per_sec


def _identify_wraps(vsig: Iterable, *,
                    min_threshold: float = 1.5,
                    max_threshold: float = 3.5):
    """
    Identify "wraps" in the voltage signal. In practice, this is when
    the encoder voltage signal crosses 5V and wraps to 0V, or
    vice-versa.

    Argument defaults and implementation suggestion via @dougo

    Parameters
    ----------
    vsig: Iterable (array-like)
        1d array-like iterable of voltage signal
    min_threshold: float (default=1.5)
        The min_threshold value that must be crossed to be considered
        a possible wrapping point.
    max_threshold: float (default=3.5)
        The max threshold value that must be crossed to be considered
        a possible wrapping point.

    Returns
    -------
    Tuple
        Tuple of ([indices of positive wraps], [indices of negative wraps])
    """
    # Compare against previous value
    shifted_vsig = _shift(vsig)
    if not isinstance(vsig, np.ndarray):
        vsig = np.array(vsig)
    # Suppress warnings for when comparing to nan values
    with np.errstate(invalid='ignore'):
        pos_wraps = np.asarray(
            np.logical_and(vsig < min_threshold, shifted_vsig > max_threshold)
            ).nonzero()[0]
        neg_wraps = np.asarray(
            np.logical_and(vsig > max_threshold, shifted_vsig < min_threshold)
            ).nonzero()[0]
    return pos_wraps, neg_wraps


def _local_boundaries(time, index, span: float = 0.25) -> tuple:
    """
    Given a 1d array of monotonically increasing timestamps, and a
    point in that array (`index`), compute the indices that form the
    inclusive boundary around `index` for timespan `span`.

    Values in `time` must monotonically increase. Flat lines (same value
    multiple times) are OK. The neighborhood may terminate around the
    index if the `span` is too small for the sampling rate. A warning
    will be raised in this case.

    Returns
    -------
    Tuple
        Tuple of corresponding to the start, end indices that bound
        a time span of length `span` (maximally)

    E.g.
    ```
    time = np.array([0, 1, 1.5, 2, 2.2, 2.5, 3, 3.5])
    _local_boundary(time, 3, 1.0)
    >>> (1, 6)
    ```
    """
    if np.diff(time[~np.isnan(time)]).min() < 0:
        raise ValueError("Data do not monotonically increase. This probably "
                         "means there is an error in your time series.")
    t_val = time[index]
    max_val = t_val + abs(span)
    min_val = t_val - abs(span)
    eligible_indices = np.nonzero((time <= max_val) & (time >= min_val))[0]
    max_ix = eligible_indices.max()
    min_ix = eligible_indices.min()
    if (min_ix == index) or (max_ix == index):
        warnings.warn("Unable to find two data points around index "
                      f"for span={span} that do not include the index. "
                      "This could mean that your time span is too small for "
                      "the time data sampling rate, the data are not "
                      "monotonically increasing, or that you are trying "
                      "to find a neighborhood at the beginning/end of the "
                      "data stream.")
    return min_ix, max_ix


def _clip_speed_wraps(speed, time, wrap_indices, t_span: float = 0.25):
    """
    Correct for artifacts at the voltage 'wraps'. Sometimes there are
    transient spikes in speed at the 'wrap' points. This doesn't make
    sense since speed on a running wheel should be a smoothly varying
    function. Take the neighborhood of values in +/- `t_span` seconds
    around wrap points, and clip the value at the wrap point
    such that it does not exceed the min/max values in the neighborhood.
    """
    corrected_speed = speed.copy()
    for wrap in wrap_indices:
        start_ix, end_ix = _local_boundaries(time, wrap, t_span)
        local_slice = np.concatenate(       # Remove the wrap point
            (speed[start_ix:wrap], speed[wrap+1:end_ix+1]))
        corrected_speed[wrap] = np.clip(
            speed[wrap], np.nanmin(local_slice), np.nanmax(local_slice))
    return corrected_speed


def _unwrap_voltage_signal(
        vsig: Iterable,
        pos_wrap_ix: Iterable,
        neg_wrap_ix: Iterable,
        *,
        vmax: Optional[float] = None,
        max_threshold: float = 5.1,
        max_diff: float = 1.0) -> np.ndarray:
    """
    Calculate the change in voltage at each timestamp.
    'Unwraps' the
    voltage data coming from the encoder at the value `vmax`. If `vmax`
    is a float, use that value to 'wrap'. If it is None, then compute
    the maximum value from the observed voltage signal (`vsig`, as long
    as the maximum value is under the value of `max_threshold` (to
    account for possible outlier data/encoder errors).
    The reason is because the rotary encoder should theoretically wrap
    at 5V, but in practice does not always reach 5V before wrapping
    back to 0V. If it is assumed that the encoder wraps at 5V, but
    actually does not reach that voltage, then the computed running
    speed can be transiently higher at the timestamps of the signal
    'wraps'.

    Parameters
    ----------
    vsig: Iterable (array-like)
        The raw voltage data from the rotary encoder
    vmax: Optional[float] (default=None)
        The value at which, upon passing this threshold, the voltage
         "wraps" back to 0V on the encoder.
    max_threshold: float (default=5.1)
        The maximum threshold for the `vmax` value. Used only if
        `vmax` is `None`. To account for the possibility of outlier
        data/encoder errors, the computed `vmax` should not exceed
        this value.
    max_diff: float (default=1.0)
        The maximum voltage difference allowed between two adjacent
        points, after accounting for the voltage "wrap". Values
        exceeding this threshold will be set to np.nan.
    Returns
    -------
    np.ndarray
        1d np.ndarray of the "unwrapped" signal from `vsig`.
    """
    if not isinstance(vsig, np.ndarray):
        vsig = np.array(vsig)
    if vmax is None:
        vmax = vsig[vsig < max_threshold].max()
    unwrapped_diff = np.zeros(vsig.shape)
    vsig_last = _shift(vsig)
    if len(pos_wrap_ix):
        # positive wraps: subtract from the previous value and add vmax
        unwrapped_diff[pos_wrap_ix] = (
            (vsig[pos_wrap_ix] + vmax) - vsig_last[pos_wrap_ix])
    # negative: subtract vmax and the previous value
    if len(neg_wrap_ix):
        unwrapped_diff[neg_wrap_ix] = (
            vsig[neg_wrap_ix] - (vsig_last[neg_wrap_ix] + vmax))
    # Other indices, just compute straight diff from previous value
    wrap_ix = np.concatenate((pos_wrap_ix, neg_wrap_ix))
    other_ix = np.array(list(set(range(len(vsig_last))).difference(wrap_ix)))
    unwrapped_diff[other_ix] = vsig[other_ix] - vsig_last[other_ix]
    # Correct for wrap artifacts based on allowed `max_diff` value
    # (fill with nan)
    # Suppress warnings when comparing with nan values to reduce noise
    with np.errstate(invalid='ignore'):
        unwrapped_diff = np.where(
            np.abs(unwrapped_diff) <= max_diff, unwrapped_diff, np.nan)
    # Get nan indices to propogate to the cumulative sum (otherwise
    # treated as 0)
    unwrapped_nans = np.array(np.isnan(unwrapped_diff)).nonzero()
    summed_diff = np.nancumsum(unwrapped_diff) + vsig[0]    # Add the baseline
    summed_diff[unwrapped_nans] = np.nan
    return summed_diff


def _zscore_threshold_1d(data: np.ndarray,
                         threshold: float = 5.0) -> np.ndarray:
    """
    Replace values in 1d array `data` that exceed `threshold` number
    of SDs from the mean with NaN.
    Parameters
    ---------
    data: np.ndarray
        1d np array of values
    threshold: float (default=5.0)
        Z-score threshold to replace with NaN.
    Returns
    -------
    np.ndarray (1d)
        A copy of `data` with values exceeding `threshold` SDs from
        the mean replaced with NaN.
    """
    corrected_data = data.copy().astype("float")
    scores = zscore(data, nan_policy="omit")
    # Suppress warnings when comparing to nan values to reduce noise
    with np.errstate(invalid='ignore'):
        corrected_data[np.abs(scores) > threshold] = np.nan
    return corrected_data


def get_running_df(
    data, time: np.ndarray, lowpass: bool = True, zscore_threshold=10.0
):
    """
    Given the data from the behavior 'pkl' file object and a 1d
    array of timestamps, compute the running speed. Returns a
    dataframe with the raw voltage data as well as the computed speed
    at each timestamp. By default, the running speed is filtered with
    a 10 Hz Butterworth lowpass filter to remove artifacts caused by
    the rotary encoder.

    Parameters
    ----------
    data
        Deserialized 'behavior pkl' file data
    time: np.ndarray (1d)
        Timestamps for running data measurements
    lowpass: bool (default=True)
        Whether to apply a 10Hz low-pass filter to the running speed
        data.
    zscore_threshold: float
        The threshold to use for removing outlier running speeds which might
        be noise and not true signal.

    Returns
    -------
    pd.DataFrame
        Dataframe with an index of timestamps and the following
        columns:
            "speed": computed running speed
            "dx": angular change, computed during data collection
            "v_sig": voltage signal from the encoder
            "v_in": the theoretical maximum voltage that the encoder
                will reach prior to "wrapping". This should
                theoretically be 5V (after crossing 5V goes to 0V, or
                vice versa). In practice the encoder does not always
                reach this value before wrapping, which can cause
                transient spikes in speed at the voltage "wraps".
        The raw data are provided so that the user may compute their
        own speed from source, if desired.

    Notes
    -----
    Though the angular change is available in the raw data
    (key="dx"), this method recomputes the angular change from the
    voltage signal (key="vsig") due to very specific, low-level
    artifacts in the data caused by the encoder. See method
    docstrings for more detailed information. The raw data is
    included in the final output in case the end user wants to apply
    their own corrections and compute running speed from the raw
    source.
    """
    v_sig = data["items"]["behavior"]["encoders"][0]["vsig"]
    v_in = data["items"]["behavior"]["encoders"][0]["vin"]

    if len(v_in) > len(time) + 1:
        error_string = ("length of v_in ({}) cannot be longer than length of "
                        "time ({}) + 1, they are off by {}").format(
            len(v_in),
            len(time),
            abs(len(v_in) - len(time))
        )
        raise ValueError(error_string)
    if len(v_in) == len(time) + 1:
        warnings.warn(
            "Time array is 1 value shorter than encoder array. Last encoder "
            "value removed\n", UserWarning, stacklevel=1)
        v_in = v_in[:-1]
        v_sig = v_sig[:-1]

    # dx = 'd_theta' = angular change
    # There are some issues with angular change in the raw data so we
    # recompute this value
    dx_raw = data["items"]["behavior"]["encoders"][0]["dx"]
    # Identify "wraps" in the voltage signal that need to be unwrapped
    # This is where the encoder switches from 0V to 5V or vice versa
    pos_wraps, neg_wraps = _identify_wraps(
        v_sig, min_threshold=1.5, max_threshold=3.5)
    # Unwrap the voltage signal and apply correction for transient spikes
    unwrapped_vsig = _unwrap_voltage_signal(
        v_sig, pos_wraps, neg_wraps, max_threshold=5.1, max_diff=1.0)
    angular_change_point = _angular_change(unwrapped_vsig, v_in)
    angular_change = np.nancumsum(angular_change_point)
    # Add the nans back in (get turned to 0 in nancumsum)
    angular_change[np.isnan(angular_change_point)] = np.nan
    angular_speed = calc_deriv(angular_change, time)  # speed in radians/s
    linear_speed = deg_to_dist(angular_speed)
    # Artifact correction to speed data
    wrap_corrected_linear_speed = _clip_speed_wraps(
        linear_speed, time, np.concatenate([pos_wraps, neg_wraps]),
        t_span=0.25)
    outlier_corrected_linear_speed = _zscore_threshold_1d(
        wrap_corrected_linear_speed, threshold=zscore_threshold)

    # Final filtering (optional) for smoothing out the speed data
    if lowpass:
        b, a = signal.butter(3, Wn=4, fs=60, btype="lowpass")
        outlier_corrected_linear_speed = signal.filtfilt(
            b, a, np.nan_to_num(outlier_corrected_linear_speed))

    return pd.DataFrame({
        'speed': outlier_corrected_linear_speed[:len(time)],
        'dx': dx_raw[:len(time)],
        'v_sig': v_sig[:len(time)],
        'v_in': v_in[:len(time)],
    }, index=pd.Index(time, name='timestamps'))


def get_running_timestamps(h5_file):
    h5 = h5py.File(h5_file, "r")

    data = h5['data'][:]
    sync_sig = data[:, -1]
    time_stamps = data[:, 0]

    meta = eval(h5['meta'][()])

    line_labels = meta['line_labels']
    sample_freq = meta['ni_daq']['sample_rate']

    # Finding rising and falling edges
    falling_edges = {}
    rising_edges = {}

    for bit, label in enumerate(line_labels):
        if not label:
            label = 'sig_' + str(bit)

        bit_array = np.bitwise_and(sync_sig, 2 ** bit).astype(bool).astype(np.uint8)
        bit_changes = np.ediff1d(bit_array, to_begin=bit_array[0])
        falling_edges[label] = time_stamps[np.where(bit_changes == 255)]/sample_freq
        rising_edges[label] = time_stamps[np.where(bit_changes == 1)]/sample_freq
    return rising_edges['vsync_stim']


def get_running_df(pkl_file, running_timestamps):
    stim_info = pd.read_pickle(pkl_file)
    zscore_threshold=10.0
    encoders = stim_info['items']['foraging']['encoders'][0]
    v_sig = encoders["vsig"]
    v_in = encoders["vin"]

    if len(v_in) > len(running_timestamps) + 1:
        error_string = ("length of v_in ({}) cannot be longer than length of "
                        "time ({}) + 1, they are off by {}").format(
            len(v_in),
            len(running_timestamps),
            abs(len(v_in) - len(running_timestamps))
        )
        raise ValueError(error_string)
    if len(v_in) == len(running_timestamps) + 1:
        warnings.warn(
            "Time array is 1 value shorter than encoder array. Last encoder "
            "value removed\n", UserWarning, stacklevel=1)
        v_in = v_in[:-1]
        v_sig = v_sig[:-1]


    dx_raw = encoders["dx"]
    # Identify "wraps" in the voltage signal that need to be unwrapped
    # This is where the encoder switches from 0V to 5V or vice versa
    pos_wraps, neg_wraps = _identify_wraps(
        v_sig, min_threshold=1.5, max_threshold=3.5)
    # Unwrap the voltage signal and apply correction for transient spikes
    unwrapped_vsig = _unwrap_voltage_signal(
        v_sig, pos_wraps, neg_wraps, max_threshold=5.1, max_diff=1.0)
    angular_change_point = _angular_change(unwrapped_vsig, v_in)
    angular_change = np.nancumsum(angular_change_point)
    # Add the nans back in (get turned to 0 in nancumsum)
    angular_change[np.isnan(angular_change_point)] = np.nan
    angular_speed = calc_deriv(angular_change, running_timestamps)  # speed in radians/s
    linear_speed = deg_to_dist(angular_speed)
    # Artifact correction to speed data
    wrap_corrected_linear_speed = _clip_speed_wraps(
        linear_speed, running_timestamps, np.concatenate([pos_wraps, neg_wraps]),
        t_span=0.25)
    outlier_corrected_linear_speed = _zscore_threshold_1d(
        wrap_corrected_linear_speed, threshold=zscore_threshold)

    # Final filtering (optional) for smoothing out the speed data

    b, a = signal.butter(3, Wn=4, fs=60, btype="lowpass")
    outlier_corrected_linear_speed = signal.filtfilt(
        b, a, np.nan_to_num(outlier_corrected_linear_speed))

    running_speed_df = pd.DataFrame({
            'timestamps': running_timestamps,
            'speed': outlier_corrected_linear_speed[:len(running_timestamps)],
        })
    speed_ = np.abs(running_speed_df.speed)
    running_speed_df['speed_filtered'] = speed_.rolling(window=50).mean()
    return running_speed_df

def resample_dff_for_trial(dff_data, dff_timestamps, t_start, t_stop,
                           resample_rate=10,
                           pre_stim=1, post_stim=1):
    """
    Resample dff traces for a single stimulus trial.

    Parameters
    ----------
    dff_data        : (n_timepoints, n_cells) raw dff matrix
    dff_timestamps  : (n_timepoints,)         timestamps of the recording
    t_start, t_stop : stimulus start / stop times (seconds)
    resample_rate   : target sampling rate (Hz)
    pre_stim        : seconds to include before stimulus onset
    post_stim       : seconds to include after stimulus offset

    Returns
    -------
    dff_resampled  : (n_new_timepoints, n_cells)  float32
    time_relative  : (n_new_timepoints,)           seconds relative to stimulus onset
    """
    window_start = t_start - pre_stim
    window_end   = t_stop  + post_stim
    dt = 1.0 / resample_rate
    time_absolute = np.arange(window_start, window_end + dt / 2, dt)
    time_relative = time_absolute - t_start

    f = interp1d(dff_timestamps, dff_data, axis=0,
                 bounds_error=False, fill_value=np.nan)
    dff_resampled = f(time_absolute)
    return dff_resampled.astype(np.float32), time_relative




def resample_running_for_trial(running_speed_df, t_start, t_stop,
                               resample_rate=10,
                               pre_stim=1, post_stim=1):
    """
    Resample running speed and speed_filtered for a single stimulus trial.

    Returns
    -------
    running_resampled : (n_timepoints, 2) float32  — columns: [speed, speed_filtered]
    time_relative     : (n_timepoints,)            seconds relative to stimulus onset
    """
    window_start = t_start - pre_stim
    window_end   = t_stop  + post_stim
    dt = 1.0 / resample_rate
    time_absolute = np.arange(window_start, window_end + dt / 2, dt)
    time_relative = time_absolute - t_start

    run_ts = running_speed_df['timestamps'].values
    f_speed = interp1d(run_ts, running_speed_df['speed'].values,
                       bounds_error=False, fill_value=np.nan)
    f_filt  = interp1d(run_ts, running_speed_df['speed_filtered'].values,
                       bounds_error=False, fill_value=np.nan)
    running_resampled = np.column_stack([
        f_speed(time_absolute),
        f_filt(time_absolute),
    ]).astype(np.float32)
    return running_resampled, time_relative

    print("Helper functions defined ✓")


def update_stimulus_table(stimulus_table):
    stimulus_table_check = stimulus_table.copy()
    stimulus_table_check = stimulus_table_check[
        ~(
            stimulus_table_check['stim_name'].astype(str).str.lower().eq('spontaneous')
            & ((stimulus_table_check['stop_time'] - stimulus_table_check['start_time']) < 2)
        )
    ].copy().reset_index(drop=True)
    stimulus_table_check.loc[stimulus_table_check['stim_name'].astype(str).str.lower().eq('spontaneous'), 'stop_time'] = \
        stimulus_table_check.loc[stimulus_table_check['stim_name'].astype(str).str.lower().eq('spontaneous'), 'stop_time'] -1
    stimulus_table_check['gray_screen'] = stimulus_table_check['contrast'].isna()
    stimulus_table_check.loc[stimulus_table_check['stim_name'].astype(str).str.lower().eq('spontaneous'), 'stim_type'] = 'GrayScreen'
    stimulus_table_check.loc[
        stimulus_table_check['gray_screen'] & stimulus_table_check['stim_type'].ne('GrayScreen'),
        'stim_type'
    ] = 'Catch'

    stimulus_table_check['stim_index'] = pd.Series(np.nan, index=stimulus_table_check.index)
    non_gray_mask = ~stimulus_table_check['gray_screen']
    stimulus_table_check.loc[non_gray_mask, 'stim_index'] = np.arange(non_gray_mask.sum(), dtype=int)

    stimulus_table_check['stim_index_block'] = pd.Series(np.nan, index=stimulus_table_check.index)
    stimulus_table_check.loc[non_gray_mask, 'stim_index_block'] = (
        stimulus_table_check.loc[non_gray_mask]
        .groupby('stim_block')
        .cumcount()
        .to_numpy()
    )

    rows_with_gaps = []
    for i in range(len(stimulus_table_check)):
        curr_row = stimulus_table_check.iloc[i].copy()
        rows_with_gaps.append(curr_row)

        if i < len(stimulus_table_check) - 1:
            next_row = stimulus_table_check.iloc[i + 1]
            if next_row['start_time'] > curr_row['stop_time']:
                gap_row = next_row.copy()
                gap_row['start_time'] = curr_row['stop_time']
                gap_row['stop_time'] = next_row['start_time']
                gap_row['gray_screen'] = True
                rows_with_gaps.append(gap_row)

    if rows_with_gaps:
        stimulus_table_check = pd.DataFrame(rows_with_gaps).reset_index(drop=True)

    stimulus_table_check['duration'] = stimulus_table_check['stop_time'] - stimulus_table_check['start_time']

    return stimulus_table_check

