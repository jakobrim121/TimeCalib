# This file provides functions that allow one to prepare the WCTE LED data for calibration

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import uproot
import sys
import json

sys.path.insert(0, "../../Geometry")
sys.path.insert(0, "../TimeCal/TimeCal")

from Geometry.WCD import WCD

from TC_Simulator import *
from TC_Data import *
from TC_Calibrator import *
from TC_Multilaterator import *

# Give us the ability to only save and not show plots
plt.ioff()  

# Written by Jakob Rimmer



def get_precal_data(data, skip_paths = [[999,999,999,999]]):
    '''
    This function takes in a dictionary and returns the pre-calibration data for each LED-PMT path.
    This function assumes that each event in the dictionary contains all the information needed to add to the TC data format.
    
    '''
    
    bad_list = [] # If you would like to exclude any mPMT-PMT pairs 
   
    bad_list_pos = skip_paths
   
    amp_threshold = 20 # Set the amplitude threshold for excluding events
    
    
    # Decide if you'd like to save the timing distributions of PMTs with clear ADC issues and/or all PMTs
    plot_double_peaks = False
    plot_all = False
    
    
    # In this version of the script, we want to extract the information on an mPMT-by-mPMT basis 
    # instead of doing it on a PMT by PMT basis...
    
    # Cycle through the events, organize the data by mPMT slot number, the transmitting mPMT and the LED position...
    data_by_slot = {}
    
    # Import the files that are needed to get the mPMT, LED, and PMT mapping
    with open('led_mapping.json', 'r') as file:
        led_mapping = json.load(file)
    
    with open('PMT_Mapping.json', 'r') as file2:
        pmt_mapping = json.load(file2)
        
    # The above list of bad mPMTs and PMTs are the card IDs and channel IDs...convert those to positions 
    # Note that "bad_list" may be empty, in which case nothing happens at this step
    
        
    for i in range(len(bad_list)):
        mpmt_slot = led_mapping[str(bad_list[i][0])]['slot_id']
        pmt_position = pmt_mapping['mapping'][str(bad_list[i][0]*100+ bad_list[i][1])] - led_mapping[str(bad_list[i][0])]['slot_id']*100
        bad_list_pos.append([mpmt_slot,pmt_position])
        
        
    # There should be no LED number 4's, so if there are, just label them to be LED number 3
    mpmt_tran_slot = led_mapping[str(data[0]['card_id'])]['slot_id']
    led_no = data[0]['led_no']
    if led_no == 4:
        led_no = 3
    led_pos = led_mapping[str(data[0]['card_id'])]['led'+str(led_no)+'_pos_id']
    
    print('mPMT slot that is firing: ' + str(mpmt_tran_slot) + ', with LED position: ' + str(led_pos) + ', flashes: ' + str(len(data)))
    
    '''
    # Set up the data_by_slot dictionary on an mPMT-by-mPMT basis
    for i in range(len(data[0])):
        for mpmt in data[i]['mpmts'].keys():
            mpmt_rec = led_mapping[str(mpmt)]['slot_id']
     
            data_by_slot['mpmt_rec' + str(mpmt_rec)] = {}
    
    for i in range(len(data[0])):
        for mpmt in data[i]['mpmts'].keys():
            mpmt_rec = led_mapping[str(mpmt)]['slot_id']
            
            for j in range(len(data[i]['mpmts'][mpmt])):
                
                try:
                    
                    pmt_id = pmt_mapping['mapping'][str(mpmt*100+data[i]['mpmts'][mpmt][j]['chan'])] - led_mapping[str(mpmt)]['slot_id']*100
                    
                except:
                    continue
              
                data_by_slot['mpmt_rec' + str(mpmt_rec)]['pmt_id' + str(pmt_id)] = {'pmt_times':[],'t_led':[],'pmt_course':[], 'wf_times':[]}
    '''
 
    
    # Now get all the PMT hit times relative to the LED flash times
    
    for i in range(len(data)):
        
        t_led = data[i]['coarse']
        mpmt_tran = data[i]['card_id']
       
        for mpmt in data[i]['mpmts'].keys():
            
            mpmt_rec = led_mapping[str(mpmt)]['slot_id']
            
            if str('mpmt_rec' + str(mpmt_rec)) not in data_by_slot:
     
                data_by_slot['mpmt_rec' + str(mpmt_rec)] = {}
        
        
            for j in range(len(data[i]['mpmts'][mpmt])):
                
                try:

                    pmt_id = pmt_mapping['mapping'][str(mpmt*100+data[i]['mpmts'][mpmt][j]['chan'])] - led_mapping[str(mpmt)]['slot_id']*100
                
                    pmt_times = data[i]['mpmts'][mpmt][j]['t'] + data[i]['mpmts'][mpmt][j]['coarse'] - t_led
                    pmt_coarse = data[i]['mpmts'][mpmt][j]['coarse']
                    wf_times = data[i]['mpmts'][mpmt][j]['t']
                    amp = data[i]['mpmts'][mpmt][j]['amp']
                
                except:
                    continue
                    
                if str('pmt_id' + str(pmt_id)) not in data_by_slot['mpmt_rec' + str(mpmt_rec)]:
                    data_by_slot['mpmt_rec' + str(mpmt_rec)]['pmt_id' + str(pmt_id)] = {'pmt_times':[],'t_led':[],'pmt_course':[], 'wf_times':[]}
                    
            
                if wf_times >0 and amp > amp_threshold:
            
                    try:
                        data_by_slot['mpmt_rec'+str(mpmt_rec)]['pmt_id' + str(pmt_id)]['pmt_times'].append(pmt_times)
                        data_by_slot['mpmt_rec'+str(mpmt_rec)]['pmt_id' + str(pmt_id)]['t_led'].append(t_led)
                        data_by_slot['mpmt_rec'+str(mpmt_rec)]['pmt_id' + str(pmt_id)]['pmt_course'].append(pmt_coarse)
                        data_by_slot['mpmt_rec'+str(mpmt_rec)]['pmt_id' + str(pmt_id)]['wf_times'].append(wf_times)
                    
                    except:
                        pass
 
    precal_data = []
    
    
        #print(data_by_slot['mpmt_rec2']['pmt_id8']['pmt_times'])
    
    # Cycle through each mPMT slot ID, extract the gaussian fit to the timing distribution
    # for PMTs that are not dead and do not have and ADC issue
    for mpmt_receiving in data_by_slot:
        mpmt_transmitting = 'mpmt_tran'+str(mpmt_tran_slot)
        led_position = 'led_pos'+str(led_pos)
        
        mpmt_rec_slot = int(mpmt_receiving[8:])
        
        # Don't do calibration on PMTs inside the mPMT that is firing the LED
        if mpmt_rec_slot == mpmt_tran_slot:
            continue
        
        gauss_fit = findNPeaks(mpmt_receiving,mpmt_transmitting,led_position, data_by_slot[mpmt_receiving], plot_double_peaks, plot_all)[1]
        
        # If there's no timing data for PMT, skip its calibration
        for pmt_id in gauss_fit[mpmt_receiving][mpmt_transmitting][led_position]:
            if len(gauss_fit[mpmt_receiving][mpmt_transmitting][led_position][pmt_id]) < 2:
                continue
                
            else:
                pmt_pos = int(pmt_id[6:])
                
                light_path = np.array([[mpmt_tran_slot,led_pos,mpmt_rec_slot,pmt_pos]])
                
                # Exclude bad paths
                if max(np.unique(np.concatenate((skip_paths,light_path)),axis=0,return_counts=True)[1])>1:
                    
                    continue
               
                fit_amp = gauss_fit[mpmt_receiving][mpmt_transmitting][led_position][pmt_id]['amp']
                t = gauss_fit[mpmt_receiving][mpmt_transmitting][led_position][pmt_id]['mu']
                t_sig = gauss_fit[mpmt_receiving][mpmt_transmitting][led_position][pmt_id]['sig']
        
                # Get the dt signal times
                dt = t*8. # convert to ns
                        
                # Get the total number of flashes    
                #n_flashes = len(data_by_slot[mpmt_receiving][pmt_id]['pmt_times'])
                n_flashes = len(data)
                
                
               
                
                fine_bin_width = 0.05
                
                # Find total number of PMT pulses caused by LED flashing
                n_gauss = np.sqrt(2*np.pi)*t_sig*fit_amp/fine_bin_width
                
                
                sig_per_flash = n_gauss/n_flashes
                
               # if mpmt_rec_slot == 52 and mpmt_tran_slot == 23:
                #    print('PMT ' + str(pmt_pos) + ', nflashes = ' + str(n_flashes) + ', sig per flash = ' + str(sig_per_flash))
                
                # Only include events that get enough light (but are not completely flooded with light)
                if sig_per_flash <0.04 or sig_per_flash > 0.8:
                    continue
                
                # Don't include fits in the calibration with sigmas>0.13
                if round(t_sig,3)>=0.13 or round(t_sig,3) <=0.04:
                    continue
        
                
                precal_data.append([mpmt_tran_slot, led_pos, mpmt_rec_slot, pmt_pos, dt, t_sig])
               
   
    
    return precal_data






def findNPeaks(mpmt_rec, mpmt_tran, led_pos, pmt_data, plot_double_peaks = True, plot_all = False):
    
    """
    Takes in the mPMT slot number and a dictionary corresponding to each of its PMTs (containing timing information etc.), and 
    returns the gaussian fitted times for PMTs that have no discernable issues (e.g. no ADC clock issue)
    
    
    """
    
    plot_path = './'
    
    eps = 0.2 # This is the range around 1 cc tick within which two peaks will be identified as a clock issue
    
    gauss_fit_t = {mpmt_rec : {}} # Place to store the gaussian fit times and standard deviations
    gauss_fit_t[mpmt_rec][mpmt_tran] = {}
    gauss_fit_t[mpmt_rec][mpmt_tran][led_pos] = {}
    
    gauss_fit_params = {mpmt_rec : {}} # Place to store all gaussian fit parameters
    gauss_fit_params[mpmt_rec][mpmt_tran] = {}
    gauss_fit_params[mpmt_rec][mpmt_tran][led_pos] = {}
    
    pmt_no_fit = {mpmt_rec : {}}
    pmt_no_fit[mpmt_rec][mpmt_tran] = {}
    pmt_no_fit[mpmt_rec][mpmt_tran][led_pos] = {}
    dead_chans = [] # List to store dead PMTs
    
    
    plot_scale = {mpmt_rec : {}} # The x-limits need to be different for each plot, store x-limits here
    adc_groups = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19]] # ADC issue comes in groups
    
    pmtid_peaks = {} # Dictionary to store number of peaks detected in each channel
    
    
    
    ### BEGIN PMT LOOP ###
    for pmt_id in pmt_data:
        plot = False
        no_hits = False
        too_few_hits = False
        no_fit = False
        title = mpmt_rec + '_' + mpmt_tran + '_' + led_pos + '_' + pmt_id
        
        cfd_fits = pmt_data[pmt_id]['pmt_times']
        plot_scale[mpmt_rec][pmt_id] = []
        gauss_fit_t[mpmt_rec][mpmt_tran][led_pos][pmt_id] = {}
        gauss_fit_params[mpmt_rec][mpmt_tran][led_pos][pmt_id] = {}
        
        
        if len(cfd_fits) == 0:
            no_hits = True
            dead_chans.append(pmt_id)
            #print('No hits present after CFD for '+ title)
            continue
        
            
       
        
        hist, bins = np.histogram(cfd_fits, bins=range(-300,500))
        
        # If there are too few hits, skip this path
        if np.sum(hist)<100:
            pmt_no_fit[mpmt_rec][mpmt_tran][led_pos][pmt_id] = True
            continue
            
        max_bin = np.argmax(hist)
        max_bin_centre = bins[max_bin] + 0.5
        # limited range of values to fit and plot
        min_val = max_bin_centre - 3.
        
        max_val = max_bin_centre + 7.
        
        # bin the data in this finer range
        fine_bin_width = 0.05

        fine_bins = np.arange(min_val,max_val,fine_bin_width)
        x, bins = np.histogram(cfd_fits, bins=fine_bins)
        #print(x)
        #print('')
        #print(x)
        
        
        # find the bin with the most entries
        max_bin = np.argmax(x)
        max_bin_center = bins[max_bin] + 0.5*fine_bin_width
        # find the mean and std deviation near this peak
        
       
        
        mod_length = max_val-min_val # Total fitting range
        nbins = len(fine_bins)
        
        # Get histogram of pulse times
        #x = np.histogram(arr,bins = nbins, range = (0,mod_length))
        
        # Find number of peaks in pulse histogram. Set threshold for peak classification (default is 20% largest peak)
        npeaks = find_peaks(x,height=np.max(x)/2,width=0.3)
        
        if len(npeaks[1]['peak_heights']) <1:
            #print(npeaks)
            continue
       
        
        # Get peak height of maximum peak, as well as the cc of this peak
        peak_max = max(npeaks[1]['peak_heights'])
        peak_max_idx = np.argmax(npeaks[1]['peak_heights'])
        peak_max_cc = min_val + npeaks[0][peak_max_idx]*(mod_length/nbins)
        
        
        num_peaks_adc = 0
        # Make sure the PMT is seeing enough hits to form a reasonable distribution to fit to
        if peak_max > 20:
            
            # Two peaks must be with 1 +- eps cc ticks in order to be classified as a clock problem
            for peak_idx in npeaks[0]: 
               
                
                # Skip the reference peak
                if peak_idx == npeaks[0][peak_max_idx]:
                    num_peaks_adc += 1
               
                
                # Compare peaks with reference peak (reference peak is largest (peak)
                elif abs(peak_max_cc-(min_val+peak_idx*(mod_length/nbins)))<1+eps and abs(peak_max_cc-(min_val+peak_idx*(mod_length/nbins)))>1-eps:
                    
                    num_peaks_adc += 1
                    plot = True
              
                    
            pmtid_peaks[pmt_id] = num_peaks_adc
        
                    
            #print('Number of peaks of interest for brb'+str(brb)+' channel ' + str(ch)+':', num_peaks_adc)
        
        #If there are very few hits, don't do any fitting
        else:
            
            num_peaks = 0
            too_few_hits = True
            pmtid_peaks[pmt_id] = 0
            
        # If there are no peaks then skip the path
        if len(npeaks[0])<1: 
            
            pmt_no_fit[mpmt_rec][mpmt_tran][led_pos][pmt_id] = True
            continue
            
      
        # Save/print plots of channels with more than one peak
        if plot_double_peaks == True:
            if (plot):
                fig = plt.figure()
                plt.hist(cfd_fits, bins = fine_bins)
                plt.xlabel('Time (8 ns bins)')
                plt.title(title +' : Peaks = ' +str(num_peaks_adc))
               
                plt.savefig(title + '_peaks' + str(num_peaks_adc))
                plt.close(fig)
                #plt.show()
                
        
            if peak_max <10: 
                fig = plt.figure()
                plt.hist(cfd_fits, bins = nbins)
                plt.xlabel('Time (8 ns bins)')
                plt.xlim(0,mod_length)
                plt.title(title)
                plt.savefig(title + '_peaks' + str(num_peaks_adc))
                plt.close(fig)
                #plt.show()
                
        # Do the gaussian fit to the distribution of times and save the mean times and standard deviations...
        
        
        
        if too_few_hits == True:
            fit_params = None
            no_fit = True
        
        else:
            try:
                # Get the fit parameters
                fit_params = get_fit(cfd_fits)
                
            except:
                
                fit_params = None
                no_fit = True
                
        pmt_no_fit[mpmt_rec][mpmt_tran][led_pos][pmt_id] = no_fit
             
        # If we couldn't perform the fit then skip the path   
        if type(fit_params) != tuple:
            
            continue;
        
        gauss_fit_params[mpmt_rec][mpmt_tran][led_pos][pmt_id]['amp'] = fit_params[0]
        gauss_fit_params[mpmt_rec][mpmt_tran][led_pos][pmt_id]['mu'] = fit_params[1]
        gauss_fit_params[mpmt_rec][mpmt_tran][led_pos][pmt_id]['sig'] = fit_params[2]
        gauss_fit_params[mpmt_rec][mpmt_tran][led_pos][pmt_id]['fit_time_bins'] = fit_params[3]
        
        # If there's no ADC issue, get timing information
        #if num_peaks_adc == 1:
        gauss_fit_t[mpmt_rec][mpmt_tran][led_pos][pmt_id]['amp'] = fit_params[0]
        gauss_fit_t[mpmt_rec][mpmt_tran][led_pos][pmt_id]['mu'] = fit_params[1]
        gauss_fit_t[mpmt_rec][mpmt_tran][led_pos][pmt_id]['sig'] = fit_params[2]
        
    ### END PMT LOOP ###
        
    # If needed, plot all timing distributions for the brb with the gaussian fits
    title_all = 'Receiving mPMT slot: ' + mpmt_rec[8:] + ' - Transmitting mPMT slot: ' + mpmt_tran[9:] + ' - LED position: '+led_pos[7:]
    
    
    if plot_all == True:
        
       
        fig, axes = plt.subplots(5,4,figsize = (23,17))
        fig.subplots_adjust(top=0.8)
        plt.suptitle(title_all)
        
        
        for pmt_id in pmt_data:
            
            hist, bins = np.histogram(pmt_data[pmt_id]['pmt_times'], bins=range(-300,500))
            max_bin = np.argmax(hist)
            max_bin_centre = bins[max_bin] + 0.5
            # Limited range of values to fit and plot
            min_val = max_bin_centre - 3.
            max_val = max_bin_centre + 7.
            
            # Bin the data in this finer range
            fine_bin_width = 0.05
            fine_bins = np.arange(min_val,max_val,fine_bin_width)
            hist, bins = np.histogram(pmt_data[pmt_id]['pmt_times'], bins=fine_bins)
            
          
            title = mpmt_rec + '_' + mpmt_tran + '_' + led_pos + '_' + pmt_id
            
            # If there's a dead PMT, just use the entire plotting range
            if np.isin(pmt_id,dead_chans) or pmt_no_fit[mpmt_rec][mpmt_tran][led_pos][pmt_id] == True:
                amp  = 0
                mu = 0
                sig = 1
                fit_time_bins = np.linspace(-300,500,1000)
                
               
            else:
                amp = round(gauss_fit_params[mpmt_rec][mpmt_tran][led_pos][pmt_id]['amp'],4)
                mu = round(gauss_fit_params[mpmt_rec][mpmt_tran][led_pos][pmt_id]['mu'],4)
                sig = round(gauss_fit_params[mpmt_rec][mpmt_tran][led_pos][pmt_id]['sig'],4)
                fit_time_bins = gauss_fit_params[mpmt_rec][mpmt_tran][led_pos][pmt_id]['fit_time_bins']
            
                
            ax = axes[int(pmt_id[6:])//4, int(pmt_id[6:])%4]
            ax.hist(pmt_data[pmt_id]['pmt_times'],bins=fine_bins)
            ax.plot(fit_time_bins,Gauss(fit_time_bins,amp,mu,sig))
            
            ax.text(0.6,0.9,'PMT position '+pmt_id[6:],transform=ax.transAxes)
            ax.text(0.6,0.8,'Mean = ' + str(round(mu,3)),transform=ax.transAxes)
            ax.text(0.6,0.7,'STD = ' + str(round(sig,3)),transform=ax.transAxes)
            ax.set_xlabel('Time (8 ns)')
           
            
                
        plt.tight_layout()
        fig.savefig('./plots/4/'+title_all+'_full_figure.png')
        plt.close(fig)
        
            
    return pmtid_peaks, gauss_fit_t


def get_fit(t0):
    
    
    hist, bins = np.histogram(t0, bins=range(-300,500))
    max_bin = np.argmax(hist)
    max_bin_centre = bins[max_bin] + 0.5
    # limited range of values to fit and plot
    min_val = max_bin_centre - 3.
    max_val = max_bin_centre + 7.
    # bin the data in this finer range
    fine_bin_width = 0.05

    fine_bins = np.arange(min_val,max_val,fine_bin_width)
    hist, bins = np.histogram(t0, bins=fine_bins)
    
    
    # find the bin with the most entries
    max_bin = np.argmax(hist)
    max_bin_center = bins[max_bin] + 0.5*fine_bin_width
    # find the mean and std deviation near this peak
    near_peak_data = [t0[i] for i in range(len(t0)) if max_bin_center-1 < t0[i] < max_bin_center+1]
    mean = np.mean(near_peak_data)
    std = np.std(near_peak_data)
    
    # fit the data to a Gaussian, using bins from -2 to +1 sigma
    nbin_below = int(2*std/fine_bin_width)
    nbin_above = int(std/fine_bin_width)
    # fine the bin that is closest to the mean
    mean_bin = int((mean-min_val)/fine_bin_width)
    fit_hist = hist[mean_bin-nbin_below:mean_bin+nbin_above]
    bin_centers = bins[mean_bin-nbin_below:mean_bin+nbin_above] + 0.5*fine_bin_width
    amp0 = min(100,max(fit_hist,default=0))
    
    fit_time_bins = np.arange(bin_centers[0],bin_centers[-1],0.01)
    
    popt, pcov = curve_fit(Gauss, bin_centers, fit_hist, p0=[amp0, mean, std])
    
    amp = popt[0]
    mu = popt[1]
    sig = abs(popt[2])
    
    return amp, mu, sig, fit_time_bins


    
def Gauss(x,a,mu,sig):
    '''
    Gaussian function to use in the gaussian fits\
    '''
    
    return a*np.exp(-((x-mu)**2)/(2*sig**2))



def get_peak_timebins(waveform, threshold, brb):
    # use the most frequent waveform value as the baseline
    values, counts = np.unique(waveform, return_counts=True)
    baseline = values[np.argmax(counts)]
    # baseline - waveform is positive going signal typically around 0 when there is no signal
    # threshold is the minimum positive signal above baseline

    below = (baseline - waveform[0]) <= threshold
    peak_timebins = []
    max_val = 0
    max_timebin = -1
    for i in range(len(waveform)):
        if below and (baseline - waveform[i]) > threshold:
            below = False
            max_val = 0
            max_timebin = -1
        if not below:
            if (baseline - waveform[i]) > max_val:
                max_val = baseline - waveform[i]
                max_timebin = i
            if (baseline - waveform[i]) <= threshold:
                below = True
                if brb != 3 or len(peak_timebins) == 0 or max_timebin > peak_timebins[-1] + 4:  # eliminate peaks from ringing... (for brb03)
                #if 1 == 1:
                    peak_timebins.append(max_timebin)
    return peak_timebins


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

cfd_true_t = [-0.5 + 0.05*i for i in range(21)]

def get_cfd(times, adcs):
    # Use a cfd like algorithm with delay d = 2, multiplier c = -2
    c = -2.
    d = 2
    # for cfd just use the average of the first 3 adcs as the baseline
    baseline = (adcs[0] + adcs[1] + adcs[2]) / 3.
    # the amplitude is found by adding the highest 3 adcs and subtracting the baseline
    #amp = (baseline - np.min(adcs)) / 100.
    n_largest_vals = sorted(baseline-np.array(adcs), reverse=True)[:3]
    amp = sum(n_largest_vals)
    # converting to positive going pulses
    data = [(baseline - adcs[i]) + c * (baseline - adcs[i - d]) for i in range(d, len(adcs))]
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
        #x -= 0.5703
        # apply a correction
        apply_correction = True
        offset = 5.
        delta = x - offset
        t = None
        if apply_correction:
            if cfd_raw_t[0] < delta < cfd_raw_t[-1]:
                correct_t = np.interp(delta,cfd_raw_t,cfd_true_t)
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
            amp = amp/2.118 # average correction
        else:
            correct_amp = np.interp(correct_t, cfd_true_t, amp_raw_t)
            amp /= correct_amp

    else:
        t = -999
        amp = -999

    return t, amp, baseline
