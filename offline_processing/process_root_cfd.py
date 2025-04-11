# This processes files from WCTE in the "offline_data" folder on EOS:
# /eos/experiment/wcte/data/2025_commissioning/offline_data/

# It applies a CFD algorithm (delay=2, multiplier=-2) to the waveforms to find the peak time and amplitude.
# The CFD algorithm is corrected for non-linearity and amplitude dependence.

# The output is a dictionary of dictionaries of lists:
# {event_number: {card_id: {channel_id: [{t: peak_time, amp: peak_amplitude, baseline: baseline,
# coarse: coarse_counter, hit_time: hit_pmt_time, hit_charge: hit_pmt_charge}]}}}

# Currently only one peak is fitted per waveform. So data[123][131][1][0]['t'] is the peak time of the first peak of
# event 123, card 131, channel 1.

# The output is saved as a pickle file in the "proc_files" folder.

# The script is run with two arguments: run_number and part_number

# Example:
# python process_wcte_root.py 947 0

# The output files are:
# proc_files/wcte_root_947_0.dict
# proc_files/wcte_root_947_0_bad.dict

# The first file contains the processed waveforms and the second file contains the waveforms that could not be processed.



import uproot as ur
import numpy as np
from pathlib import Path
import pickle
import sys

# cfd non-linear time correction:

cfd_raw_t = [0.16323452713658082,
             0.20385733509493395,
             0.24339187740767365,
             0.2822514122310461,
             0.3208335490313887,
             0.35953379168152044,
             0.3987592183841288,
             0.4389432980060811,
             0.4805630068163285,
             0.5241597383052767,
             0.5703660640730557,
             0.6199413381955754,
             0.6738206794685682,
             0.7331844507933303,
             0.7995598000823612,
             0.874973724581176,
             0.9621917102137131,
             1.0301530251726216,
             1.0769047405430523,
             1.1210801763323819,
             1.1632345271365807]

# correction for the amplitude derived by summing the largest three adcs
amp_raw_t = [2.0413475167493225, 2.0642014124776784, 2.0847238089021274, 2.1028869067818117, 2.118667914530039,
             2.1320484585033723, 2.1430140317025583, 2.151553497195665, 2.1576586607668613, 2.1613239251470255,
             2.162546035746829, 2.1613239251470255, 2.1576586607668617, 2.1515534971956654, 2.143014031702558,
             2.1320484585033723, 2.118667914530039, 2.1028869067818117, 2.0847238089021274, 2.0642014124776784,
             2.0413475167493225]

# bins for the above constants
cfd_true_t = [-0.5 + 0.05 * i for i in range(21)]


# The waveforms are inverted (positive peak)

def get_peak_timebins(waveform, threshold):
    # for peak finding, the baseline is assumed to be zero
    baseline = 0
    # threshold is the minimum positive signal above baseline

    below = (waveform[0] - baseline) <= threshold
    peak_timebins = []
    max_val = 0
    max_timebin = -1
    for i in range(len(waveform)):
        if below and (waveform[i] - baseline) > threshold:
            below = False
            max_val = 0
            max_timebin = -1
        if not below:
            if (waveform[i] - baseline) > max_val:
                max_val = waveform[i] - baseline
                max_timebin = i
            if (waveform[i] - baseline) <= threshold:
                below = True
                peak_timebins.append(max_timebin)
            elif i == len(waveform) - 1:
                peak_timebins.append(max_timebin)

    return peak_timebins


def get_cfd(adcs):
    # Use a cfd like algorithm with delay d = 2, multiplier c = -2
    c = -2.
    d = 2
    # for cfd just use the average of the first 3 adcs as the baseline
    baseline = (adcs[0] + adcs[1] + adcs[2]) / 3.
    # the amplitude is found by adding the highest 3 adcs and subtracting the baseline
    # amp = (baseline - np.min(adcs)) / 100.
    n_largest_vals = sorted(np.array(adcs) - baseline, reverse=True)[:3]
    amp = sum(n_largest_vals)
    # converting to positive going pulses
    data = [(adcs[i] - baseline) + c * (adcs[i - d] - baseline) for i in range(d, len(adcs))]
    # find largest swing zero crossing
    max_diff = 0
    i_md = -1
    for iv in range(1, len(data)):
        if data[iv - 1] > 0. and data[iv] < 0.:
            if data[iv - 1] - data[iv] > max_diff:
                max_diff = data[iv - 1] - data[iv]
                i_md = iv

    if i_md > -1:
        x0 = i_md - 1
        y0 = data[i_md - 1]
        x1 = i_md
        y1 = data[i_md]

        # using a linear interpolation, find the value of x for which y = 0
        x = x0 - (x1 - x0) / (y1 - y0) * y0
        # apply offset assuming sigma = 0.96 (see try_cfd.ipynb)
        # x -= 0.5703
        # apply a correction
        apply_correction = True
        offset = 5.
        delta = x - offset
        t = None
        if apply_correction:
            if cfd_raw_t[0] < delta < cfd_raw_t[-1]:
                correct_t = np.interp(delta, cfd_raw_t, cfd_true_t)
                t = offset + correct_t
            elif delta < cfd_raw_t[0]:
                delta += 1
                if cfd_raw_t[0] < delta < cfd_raw_t[-1]:
                    correct_t = np.interp(delta, cfd_raw_t, cfd_true_t)
                    t = offset - 1 + correct_t
            elif delta > cfd_raw_t[-1]:
                delta -= 1
                if cfd_raw_t[0] < delta < cfd_raw_t[-1]:
                    correct_t = np.interp(delta, cfd_raw_t, cfd_true_t)
                    t = offset + 1 + correct_t
        if t is None:
            t = x - 0.5703
            amp = amp / 2.118  # average correction
        else:
            correct_amp = np.interp(correct_t, cfd_true_t, amp_raw_t)
            amp /= correct_amp

    else:
        t = -999
        amp = -999

    return t, amp, baseline


# Process waveforms and produce a dictionary

# run_number = 947
# part_number = 0
run_number = sys.argv[1]
part_number = sys.argv[2]
srn = str(run_number)
spn = str(part_number)

print('Analyzing run ' + srn + ' part ' + spn)
from pathlib import Path

Path("/eos/user/j/jrimmer/led_data/dictionaries/tooldaq/self_trigger/" + srn + "/").mkdir(parents=True, exist_ok=True)

folder = "/eos/user/j/jrimmer/led_data/dictionaries/tooldaq/self_trigger/" + srn + "/"
run_file = ur.open(
    "/eos/experiment/wcte/data/2025_commissioning/offline_data/" + srn + "/WCTE_offline_R" + srn + "S0P" + spn + ".root")
tree = run_file['WCTEReadoutWindows']

hit_pmt_times = tree['hit_pmt_times'].array()
card_ids = tree['pmt_waveform_mpmt_card_ids'].array()
channel_ids = tree['pmt_waveform_pmt_channel_ids'].array()
led_ids = tree['led_ids'].array()
charges = tree['hit_pmt_charges'].array()
window_times = tree['window_time'].array()
trigger_types = tree['trigger_types'].array()
waveform_times = tree['pmt_waveform_times'].array()
waveforms = tree['pmt_waveforms'].array()
led_card_ids = tree['led_card_ids'].array()
trigger_times = tree['trigger_times'].array()



events = {}
bad_events = {}
for ie in range(len(card_ids)):
    event = {}
    bad_event = {}
    fit_count = 0
    bad_count = 0
    window_time_cc = window_times[ie]/8
    
    trigger_time_cc = trigger_times[ie][trigger_types[ie]==2][0]/8
    
    led_no = led_ids[ie][0]
    
    led_card = led_card_ids[ie][0]
    
    event['coarse'] = trigger_time_cc
    event['led_no'] = led_no
    event['card_id'] = led_card
    event['mpmts'] = {}
    
    for iw in range(len(card_ids[ie])):

        try:

            card_id = card_ids[ie][iw]
            channel_id = channel_ids[ie][iw]
            waveform = waveforms[ie][iw]
            waveform_time_cc = waveform_times[ie][iw]/ 8
        
            hit_pmt_time_cc = hit_pmt_times[ie][iw]/8
            
        except:
            continue


        try:
            hit_pmt_time = hit_pmt_times[ie][iw]
        except:
            hit_pmt_time = -999999999999

        try:
            hit_pmt_charge = hit_pmt_charges[ie][iw]
        except:
            hit_pmt_charge = -999999999999

        threshold = 20
        peak_timebins = get_peak_timebins(waveform, threshold)
        # look at single peak waveforms
        if 0 < len(peak_timebins) < 5:

            offset = peak_timebins[0] - 7  # where the pulse starts in the sample
            if offset > -1 and peak_timebins[0] < len(waveform) - 4:
                t, amp, baseline = get_cfd(waveform[offset:offset + 11])
                t += offset

                cfd_fit = {'chan': channel_id,
                            't': t,
                           'amp': amp,
                           'baseline': baseline,
                           'coarse': waveform_time_cc,
                           'window_time':window_time_cc,
                           'hit_time': hit_pmt_time,
                           'hit_charge': hit_pmt_charge}
                
                if card_id not in event['mpmts']:
                    event['mpmts'][card_id] = []
                
                event['mpmts'][card_id].append(cfd_fit)
                fit_count += 1
        else:
            if card_id not in bad_event:
                bad_event[card_id] = {}
            if channel_id not in bad_event[card_id]:
                bad_event[card_id][channel_id] = []
            bad_event[card_id][channel_id].append(waveform)
            bad_count += 1

    events[ie] = event
    bad_events[ie] = bad_event

    if ie % 100 == 0:
        print('Analyzed event', ie)
        print('    ', fit_count, ' peaks fit')
        print('    ', bad_count, ' bad waveforms')
        
        
try:
    filepath = Path(folder + "/wcte_root_" + srn + "_" + spn + ".dict").resolve()
except:
    raise TypeError('Input arg could not be converted to a valid path: {}' +
                    '\n It must be a str or Path-like.')

if len(filepath.suffix) < 2:
    filepath = filepath.with_suffix('.dict')

with open(filepath, 'wb') as f:
    pickle.dump(events, f, protocol=4)

try:
    filepath = Path(folder + "/wcte_root_" + srn + "_" + spn + "_bad.dict").resolve()
except:
    raise TypeError('Input arg could not be converted to a valid path: {}' +
                    '\n It must be a str or Path-like.')

if len(filepath.suffix) < 2:
    filepath = filepath.with_suffix('.dict')

with open(filepath, 'wb') as f:
    pickle.dump(bad_events, f, protocol=4)
