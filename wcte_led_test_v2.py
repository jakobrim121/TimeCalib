import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
#from peak_timebins import get_peak_timebins
from scipy.optimize import curve_fit
#from cfd import get_cfd
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

# The following functions are useful for extracting the timing calibration constants for mPMTs and PMTs in the WCTE. 

# Written by Jakob Rimmer


#def get_TC_data(run, file_path, nbins):
def get_calibration_constants(run, test_data, skip_pmts = []):
    '''
    This function takes in a rootfile and returns the calibration constants for each PMT and mPMT.
    This function assumes that each event in the rootfile contains all the information needed to add to the TC data format.
    
    '''
    
    calib_const = {}
    
    bad_list = np.array([
     [10,16],[10,17],[10,18],[21,0],[21,1],[21,2],[21,3],[21,4],[21,5],[21,6],[21,7],[26,0],[26,1],[26,2],[26,3],
    [42,4],[42,5],[42,6],[42,7],[45,0],[45,1],[45,2],[45,3],[47,4],[47,5],[47,6],[47,7],[52,4],[52,5],[52,6],[52,7],
    [78,4],[78,5],[78,6],[78,7],[83,4],[83,5],[83,6],[83,7],[85,4],[85,5],[85,6],[85,7],[93,4],[93,5],[93,6],[93,7],
    [95,0],[95,1],[95,2],[95,3],[97,4],[97,5],[97,6],[97,7]
    ])
    
    
    bad_list_pos = [[1,1],[1,2],[1,6],[1,7],[1,8],[1,18],[2,1],[2,2],[2,7],[2,8],[4,16],[5,1],[5,5],[5,6],[5,10],[5,12],[8,18],[11,17],[20,0],[20,6],[20,17],[22,1],[22,2],[22,7],[22,8],[29,0],[29,2],[29,8],[29,11],[31,1],[31,2],[31,7],[31,8],[31,14],[31,15],[31,16],[33,4],[39,12],[40,13],[42,4],[42,5],[42,12],[43,1],[43,2],[43,7],[43,8],[43,17],[53,0],[53,7],[53,8],[60,13],[61,0],[61,1],[61,2],[61,5],[65,14],[65,15],[65,16],[67,7],[78,0],[78,3],[78,4],[78,6],[78,8],[78,12],[81,14],[83,1],[83,7],[86,1],[86,2],[86,7],[86,8],[86,14],[86,15],[86,16],[88,1],[88,2],[88,7],[88,8],[88,10],[90,0],[90,7],[90,6],[90,14],[90,17],[90,18],[93,0],[93,3],[93,4],[93,6],[93,9],[93,16],[93,17],[93,18],[94,1],[94,2],[94,7],[101,0],[101,3],[101,4],[101,5],[101,6],[101,10],[101,13],[101,14],[101,15],[101,16],[101,17],[103,3],[103,11],[104,11]]
    
    amp_threshold = 20
    
    # Set up TC data structure in which to place the LED and PMT information
    wcte = WCD('wcte', kind='WCTE')
    wcte_calib = TC_Calibrator(wcte) 
    tc_led_data = TC_Data(str(run))
    
    # Decide if you'd like to save the timing distributions of PMTs with clear ADC issues and/or all PMTs
    plot_double_peaks = False
    plot_all = False
    
    data = test_data # Extract the events from the rootfile
    
    # In this version of the script, we want to extract the information on an mPMT-by-mPMT basis 
    # instead of doing it on a PMT by PMT basis...
    
    # Cycle through the events, organize the data by mPMT slot number, the transmitting mPMT and the LED position...
    data_by_slot = {}
    
    rec_list = []
    tran_list = []
    led_list = []
    rec_tran = []
    rec_tran_led = []
    
    # If there are bad PMT positions passed as an argument, then parse them and add them to the existing list
    if len(skip_pmts)>0:
        for i in range(len(skip_pmts)):
            # Add additional bad PMTs to this list
            if max(np.unique(np.concatenate((bad_list_pos,[skip_pmts[i]])),axis=0,return_counts=True)[1])==1:
                bad_pos_list.append(skip_pmts[i])
    
    
    with open('led_mapping.json', 'r') as file:
        led_mapping = json.load(file)
    
    with open('PMT_Mapping.json', 'r') as file2:
        pmt_mapping = json.load(file2)
        
    # The above list of bad mPMTs and PMTs are the card IDs and channel IDs...convert those to positions    
    #bad_list_pos = []
        
    for i in range(len(bad_list)):
        mpmt_slot = led_mapping[str(bad_list[i][0])]['slot_id']
        pmt_position = pmt_mapping['mapping'][str(bad_list[i][0]*100+ bad_list[i][1])] - led_mapping[str(bad_list[i][0])]['slot_id']*100
        bad_list_pos.append([mpmt_slot,pmt_position])
        #print('card_id' + str(str(bad_list[i][0])) + ', slot' + str(mpmt_slot) + ', chan' + str(str(bad_list[i][1])) + ', position' + str(pmt_position))
    
        
    mpmt_tran_slot = led_mapping[str(data[0]['card_id'])]['slot_id']
    led_no = data[0]['led_no']
    led_pos = led_mapping[str(data[0]['card_id'])]['led'+str(led_no)+'_pos_id']
    print('mPMT slot that is firing: ' + str(mpmt_tran_slot) + ', with LED position: ' + str(led_pos))
    
    
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
                    #print('Channel ', data[i]['mpmts'][mpmt][j]['chan'])
                    #print('Position ', pmt_id)
                    #print('')
                except:
                    continue
              
                data_by_slot['mpmt_rec' + str(mpmt_rec)]['pmt_id' + str(pmt_id)] = {'pmt_times':[],'t_led':[],'pmt_course':[], 'wf_times':[]}
 
    
    for i in range(len(data)):
        t_led = data[i]['coarse']
        mpmt_tran = data[i]['card_id']
       
        for mpmt in data[i]['mpmts'].keys():
            mpmt_rec = led_mapping[str(mpmt)]['slot_id']
        
        
            for j in range(len(data[i]['mpmts'][mpmt])):
                
                try:

                    pmt_id = pmt_mapping['mapping'][str(mpmt*100+data[i]['mpmts'][mpmt][j]['chan'])] - led_mapping[str(mpmt)]['slot_id']*100
                except:
                    continue
        
                pmt_times = data[i]['mpmts'][mpmt][j]['t'] + data[i]['mpmts'][mpmt][j]['coarse'] - t_led
                pmt_coarse = data[i]['mpmts'][mpmt][j]['coarse']
                wf_times = data[i]['mpmts'][mpmt][j]['t']
                amp = data[i]['mpmts'][mpmt][j]['amp']
            
                if wf_times >0 and amp > amp_threshold:
            
                    data_by_slot['mpmt_rec'+str(mpmt_rec)]['pmt_id' + str(pmt_id)]['pmt_times'].append(pmt_times)
                    data_by_slot['mpmt_rec'+str(mpmt_rec)]['pmt_id' + str(pmt_id)]['t_led'].append(t_led)
                    data_by_slot['mpmt_rec'+str(mpmt_rec)]['pmt_id' + str(pmt_id)]['pmt_course'].append(pmt_coarse)
                    data_by_slot['mpmt_rec'+str(mpmt_rec)]['pmt_id' + str(pmt_id)]['wf_times'].append(wf_times)
 
    fit_times = []
    to_be_calib = []
    
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
        #print(gauss_fit)
        #print('')
                
                # If there's no timing data for PMT, skip its calibration
        for pmt_id in gauss_fit[mpmt_receiving][mpmt_transmitting][led_position]:
            if len(gauss_fit[mpmt_receiving][mpmt_transmitting][led_position][pmt_id]) < 2:
                continue
                
            else:
                pmt_pos = int(pmt_id[6:])
                
                mpmt_pmt_pair = np.array([[mpmt_rec_slot,pmt_pos]])
                
                # Exclude PMTs in bad PMT list
                if max(np.unique(np.concatenate((bad_list_pos,mpmt_pmt_pair)),axis=0,return_counts=True)[1])>1:
                    #print('bad detected')
                    continue
                
                
                #pmt_coarse = data_by_slot[mpmt_receiving][pmt_id][pmt_coarse]
                fit_amp = gauss_fit[mpmt_receiving][mpmt_transmitting][led_position][pmt_id]['amp']
                t = gauss_fit[mpmt_receiving][mpmt_transmitting][led_position][pmt_id]['mu']
                t_sig = gauss_fit[mpmt_receiving][mpmt_transmitting][led_position][pmt_id]['sig']
        
                #Get the dt signal times
                dt = t*8. # convert to ns
                        
                    
                n_flashes = len(data_by_slot[mpmt_receiving][pmt_id]['pmt_times'])
               
             
                
                fine_bin_width = 0.05
                
                n_gauss = np.sqrt(2*np.pi)*t_sig*fit_amp/fine_bin_width
                
                
                sig_per_flash = n_gauss/n_flashes
                #print(sig_per_flash)
                
                if sig_per_flash <0.04 or sig_per_flash > 0.8:
                    continue
                
                # Don't include fits in the calibration with sigmas>0.14
                if round(t_sig,2)>=0.13 or round(t_sig,2) <=0.04:
                    continue
        
                tc_led_data.set(mpmt_tran_slot, led_pos, mpmt_rec_slot, pmt_pos, dt, t_sig)
                to_be_calib.append([mpmt_tran_slot, led_pos, mpmt_rec_slot, pmt_pos, dt, t_sig])
                #fit_times.append([mpmt_tran_slot, led_pos, mpmt_rec_num, pmt_id_num, t, t_sig])
   
        
    #wcte_calib.assign_data(tc_led_data)
    #wcte_calib.set_reference_mpmt(10,0,0.1)
    #wcte_calib.set_priors(0.,10.,0.,10.) 
        
    #chi2, n_dof =wcte_calib.calibrate()
    
    #my_wcte.mpmts[0].pmts[0].prop_true['delay']
    #calib_const = {'mpmt_slot_delay':[]}
    #for mpmt in wcte.mpmts:
        #calib_const[mpmt] = {}
        #calib_const[mpmt]['mpmt_delay'] = wcte.mpmts[0].prop_est
        #calib_const[mpmt] = 
       # x=1
        
    
    return to_be_calib



def plot_only(run, data, plot_double_peaks, plot_all):
    
    
    
    #plot_double_peaks = True
    #plot_all = False
    
    #events = test_data # Extract the events from the rootfile
    
    # In this version of the script, we want to extract the information on an mPMT-by-mPMT basis 
    # instead of doing it on a PMT by PMT basis...
    
    # Cycle through the events, organize the data by mPMT slot number, the transmitting mPMT and the LED position...
    data_by_slot = {}
    
    rec_list = []
    tran_list = []
    led_list = []
    rec_tran = []
    rec_tran_led = []
    
    amp_threshold = 20

    with open('led_mapping.json', 'r') as file:
        led_mapping = json.load(file)
    
    with open('PMT_Mapping.json', 'r') as file2:
        pmt_mapping = json.load(file2)
        
        
        
    mpmt_tran_slot = led_mapping[str(data[0]['card_id'])]['slot_id']
    led_no = data[0]['led_no']
    led_pos = led_mapping[str(data[0]['card_id'])]['led'+str(led_no)+'_pos_id']
    
    print('mPMT slot that is firing: ' + str(mpmt_tran_slot) + ', with LED position: ' + str(led_pos))
    
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
 

    for i in range(len(data)):
        t_led = data[i]['coarse']
        #mpmt_tran = data[i]['card_id']
       
        for mpmt in data[i]['mpmts'].keys():
            mpmt_rec = led_mapping[str(mpmt)]['slot_id']
        
        
            for j in range(len(data[i]['mpmts'][mpmt])):
                
                try:

                    pmt_id = pmt_mapping['mapping'][str(mpmt*100+data[i]['mpmts'][mpmt][j]['chan'])] - led_mapping[str(mpmt)]['slot_id']*100
                except:
                    continue
        
                pmt_times = data[i]['mpmts'][mpmt][j]['t'] + data[i]['mpmts'][mpmt][j]['coarse'] - t_led
                pmt_coarse = data[i]['mpmts'][mpmt][j]['coarse']
                wf_times = data[i]['mpmts'][mpmt][j]['t']
                amp = data[i]['mpmts'][mpmt][j]['amp']
            
                if wf_times >0 and amp > amp_threshold:
            
                    data_by_slot['mpmt_rec'+str(mpmt_rec)]['pmt_id' + str(pmt_id)]['pmt_times'].append(pmt_times)
                    data_by_slot['mpmt_rec'+str(mpmt_rec)]['pmt_id' + str(pmt_id)]['t_led'].append(t_led)
                    data_by_slot['mpmt_rec'+str(mpmt_rec)]['pmt_id' + str(pmt_id)]['pmt_course'].append(pmt_coarse)
                    data_by_slot['mpmt_rec'+str(mpmt_rec)]['pmt_id' + str(pmt_id)]['wf_times'].append(wf_times)
 
        
    for mpmt_receiving in data_by_slot:
        
        mpmt_transmitting = 'mpmt_tran'+str(mpmt_tran_slot)
        led_position = 'led_pos'+str(led_pos)
        
        if np.isin(int(mpmt_tran_slot),np.array([21,23,60])):
            plot_all = True
               
        findNPeaks(mpmt_receiving,mpmt_transmitting,led_position, data_by_slot[mpmt_receiving], plot_double_peaks, plot_all)[1]
                    
        
        
       
    return 0


def write_json(start_run_num,end_run_num,start_time,end_time,official,data):
    
    folder = "/eos/user/j/jrimmer/SWAN_projects/wcte_led/results/json_files/"
    
    '''
    Function to write out data in the correct json format, which will then be uploaded to the calibration database.
    Note: data variable should be a list of dictionaries of pmt_id's and timing offsets
    
    '''
    
    # Data to be written
    dictionary = {
    "start_run_number": start_run_num,
    "end_run_number": end_run_num,
    "start_time": start_time,
    "end_time": end_time,
    "calibration_name": "timing_offsets",
    "calibration_method": "LED",
    "official": official,
    "data": data
    }
    
    # Serializing json
    json_object = json.dumps(dictionary, indent=4)
 
    # Writing to json file
    with open(folder + "SR"+str(start_run_num)+"_ER" + str(end_run_num)+"_ST" + str(start_time)+"_ET"+str(end_time), "w") as outfile:
        outfile.write(json_object)
 
    return 0



def findNPeaks(mpmt_rec, mpmt_tran, led_pos, pmt_data, plot_double_peaks = True, plot_all = False):
    
    """
    Takes in the mPMT slot number and a dictionary corresponding to each of its PMTs (containing timing information etc.), and 
    returns the gaussian fitted times for PMTs that have no discernable issues (e.g. no ADC clock issue)
    
    
    """
    
    plot_path = './'
    
    eps = 0.2 # This is the range around 1 cc tick within which two peaks will be identified as a clock issue
    
    #data_length = 1024 # Number of cc ticks in a waveform
    #cfd_fits = {mpmt_rec : {}} # Create a dictionary for cfd fits to be placed in
    #timing_dist = {mpmt_rec : {}} # Dictionary to hold the raw CFD timing fits
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
    
    #left_fit_cc_offset = 2 # How many cc ticks to the left of the distribution to fit (where to start fit)
    #right_fit_cc_offset = 0.5 # How many cc ticks to the right of the distribution to fit (where to stop fit)
    #mod_length = cc_length 
    
    #bin_ratio = nbins/mod_length # Number of histogram bins per cc tick
    #plot_cc_offset = 3
    #offset = 3 # Offset from peaks used for plotting
    plot_scale = {mpmt_rec : {}} # The x-limits need to be different for each plot, store x-limits here
    adc_groups = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19]] # ADC issue comes in groups
    
    pmtid_peaks = {} # Dictionary to store number of peaks detected in each channel
    
    
    
    ### BEGIN PMT LOOP ###
    for pmt_id in pmt_data:
        plot = False
        no_hits = False
        too_few_hits = False
        title = mpmt_rec + '_' + mpmt_tran + '_' + led_pos + '_' + pmt_id
        
        cfd_fits = pmt_data[pmt_id]['pmt_times']
        #timing_dist[mpmt_rec][pmt_id] = []
        plot_scale[mpmt_rec][pmt_id] = []
        gauss_fit_t[mpmt_rec][mpmt_tran][led_pos][pmt_id] = {}
        gauss_fit_params[mpmt_rec][mpmt_tran][led_pos][pmt_id] = {}
        
        
        if len(cfd_fits) == 0:
            no_hits = True
            dead_chans.append(pmt_id)
            #print('No hits present after CFD for '+ title)
            continue
        
            
        arr = cfd_fits
        
        hist, bins = np.histogram(arr, bins=range(500))
        if np.sum(hist)<40:
            continue
            
        max_bin = np.argmax(hist)
        max_bin_centre = max_bin + 0.5
        # limited range of values to fit and plot
        min_val = max_bin_centre - 3.
        
        max_val = max_bin_centre + 7.
        # bin the data in this finer range
        fine_bin_width = 0.05

        fine_bins = np.arange(min_val,max_val,fine_bin_width)
        x, bins = np.histogram(arr, bins=fine_bins)
        #print(x)
        
        
        # find the bin with the most entries
        max_bin = np.argmax(x)
        max_bin_center = bins[max_bin] + 0.5*fine_bin_width
        # find the mean and std deviation near this peak
        
       
        
        mod_length = max_val-min_val
        nbins = len(fine_bins)
        
        # Get histogram of pulse times
        #x = np.histogram(arr,bins = nbins, range = (0,mod_length))
        
        # Find number of peaks in pulse histogram. Set threshold for peak classification (default is 20% largest peak)
        npeaks = find_peaks(x,height=np.max(x)/2,width=0.3)
        
        #Save the x-limits needed for the plot 
        #plot_scale[mpmt_rec][pmt_id].append([(mod_length/nbins)*npeaks[0][0]-offset,(mod_length/nbins)*npeaks[0][-1]+offset])
        
        
        
        # Get peak height of maximum peak, as well as the cc of this peak
        peak_max = max(npeaks[1]['peak_heights'])
        peak_max_idx = np.argmax(npeaks[1]['peak_heights'])
        peak_max_cc = min_val + npeaks[0][peak_max_idx]*(mod_length/nbins)
        
        #print('Peaks: ', npeaks[1]['peak_heights'])
        
        #print(peak_max_cc)
        
        num_peaks_adc = 0
        # Make sure the PMT is seeing enough hits to form a reasonable distribution to fit to
        if peak_max > 20:
            
            # Two peaks must be with 1 +- eps cc ticks in order to be classified as a clock problem
            for peak_idx in npeaks[0]: 
               
                #print(abs(peak_max_cc-(min_val+peak_idx*(mod_length/nbins))))
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
            #print('Too few hits to determine peaks for ' + title)
            num_peaks = 0
            too_few_hits = True
            pmtid_peaks[pmt_id] = 0
            
        if len(npeaks[0])<1: 
            #print('No peaks detected for ' + title)
            
            continue
            
            
         #Save the x-limits needed for the plot 
        #if num_peaks_adc > 1:
            #plot_scale[mpmt_rec][pmt_id].append([(mod_length/nbins)*npeaks[0][0]-offset,(mod_length/nbins)*npeaks[0][-1]+offset])
            
        #else:
            #plot_scale[mpmt_rec][pmt_id].append([(mod_length/nbins)*npeaks[0][peak_max_idx]-offset,(mod_length/nbins)*npeaks[0][peak_max_idx]+offset])
            
            
           
       
        # Save/print plots of channels with more than one peak
        if plot_double_peaks == True:
            if (plot):
                fig = plt.figure()
                plt.hist(arr, bins = fine_bins)
                plt.xlabel('Time (8 ns bins)')
                plt.title(title +' : Peaks = ' +str(num_peaks_adc))
                #plt.xlim((mod_length/nbins)*npeaks[0][0]-offset, (mod_length/nbins)*npeaks[0][-1]+offset)
                
                plt.savefig(title + '_peaks' + str(num_peaks_adc))
                plt.close(fig)
                #plt.show()
                
        
            if peak_max <10: 
                fig = plt.figure()
                plt.hist(arr, bins = nbins)
                plt.xlabel('Time (8 ns bins)')
                plt.xlim(0,mod_length)
                plt.title(title)
                plt.savefig(title + '_peaks' + str(num_peaks_adc))
                plt.close(fig)
                #plt.show()
                
        # Do the gaussian fit to the distribution of times and save the mean times and standard deviations...
        
        
        no_fit = False
        if too_few_hits == True:
            fit_params = None
            no_fit = True
        
        else:
            try:
                fit_params = get_fit(cfd_fits)
                
            except:
                
                fit_params = None
                no_fit = True
                
        pmt_no_fit[mpmt_rec][mpmt_tran][led_pos][pmt_id] = no_fit
             
        if type(fit_params) != tuple:
            #print('Unable to produce gaussian fit for ' + title)
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
    
    #title_all = mpmt_rec + '_' + mpmt_tran + '_' + led_pos
    if plot_all == True:
        
        #xfit = np.linspace(0,mod_length,10000)
        #x_fit_ratio = len(xfit)/mod_length # How many x points within one cc tick used for plotting the gaussian fit
        
        fig, axes = plt.subplots(5,4,figsize = (23,17))
        fig.subplots_adjust(top=0.8)
        plt.suptitle(title_all)
        #plt.suptitle('mPMT Card ID' + str(mpmt))
        
        for pmt_id in pmt_data:
            
            hist, bins = np.histogram(pmt_data[pmt_id]['pmt_times'], bins=range(500))
            max_bin = np.argmax(hist)
            max_bin_centre = max_bin + 0.5
            # limited range of values to fit and plot
            min_val = max_bin_centre - 3.
            max_val = max_bin_centre + 7.
            # bin the data in this finer range
            fine_bin_width = 0.05
            fine_bins = np.arange(min_val,max_val,fine_bin_width)
            hist, bins = np.histogram(pmt_data[pmt_id]['pmt_times'], bins=fine_bins)
            
          
            title = mpmt_rec + '_' + mpmt_tran + '_' + led_pos + '_' + pmt_id
            
            # If there's a dead PMT, just use the entire plotting range
            if np.isin(pmt_id,dead_chans) or pmt_no_fit[mpmt_rec][mpmt_tran][led_pos][pmt_id] == True:
                amp  = 0
                mu = 0
                sig = 1
                fit_time_bins = np.linspace(0,500,500)
                
                #xfit_small = xfit
                #plot_scale[mpmt_rec][pmt_id].append([0,mod_length])
                
            else:
                amp = round(gauss_fit_params[mpmt_rec][mpmt_tran][led_pos][pmt_id]['amp'],4)
                mu = round(gauss_fit_params[mpmt_rec][mpmt_tran][led_pos][pmt_id]['mu'],4)
                sig = round(gauss_fit_params[mpmt_rec][mpmt_tran][led_pos][pmt_id]['sig'],4)
                fit_time_bins = gauss_fit_params[mpmt_rec][mpmt_tran][led_pos][pmt_id]['fit_time_bins']
            
                #xfit_small = np.intersect1d(xfit[xfit>mu-left_fit_cc_offset], xfit[xfit<mu+right_fit_cc_offset])
            
            # Specify positions for the text on the plots
            #xpos1 = plot_scale[mpmt_rec][pmt_id][0][0] + 1 
            #ypos1 = amp/2
            
            #xpos2 = plot_scale[mpmt_rec][pmt_id][0][0] + 1
            #ypos2 = amp/2.5
            
            ax = axes[int(pmt_id[6:])//4, int(pmt_id[6:])%4]
            ax.hist(pmt_data[pmt_id]['pmt_times'],bins=fine_bins)
            ax.plot(fit_time_bins,Gauss(fit_time_bins,amp,mu,sig))
            
            ax.text(0.6,0.9,'PMT position '+pmt_id[6:],transform=ax.transAxes)
            ax.text(0.6,0.8,'Mean = ' + str(round(mu,3)),transform=ax.transAxes)
            ax.text(0.6,0.7,'STD = ' + str(round(sig,3)),transform=ax.transAxes)
            ax.set_xlabel('Time (8 ns)')
            #ax.set_title(title)
            #ax.text()
           
                
            
                
        plt.tight_layout()
        fig.savefig('./plots/'+title_all+'_full_figure.png')
        plt.close(fig)
        
            
    return pmtid_peaks, gauss_fit_t

def get_fit(t0):
    
    
    hist, bins = np.histogram(t0, bins=range(500))
    max_bin = np.argmax(hist)
    max_bin_centre = max_bin + 0.5
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

"""
def get_fit(t0, cc_length, left_cc_offset, right_cc_offset, nbins):
    
    '''
    Function to get the gauss fit time from the timing distribution after the CFD fit has occurred
    '''
    
    if len(t0) == 0:
        return 
    
    mod_length = cc_length
    #nbins = 2000
    ratio = nbins/mod_length
    left_off = int(left_cc_offset*ratio) # where to start fit on left of peak center
    right_off = int(right_cc_offset*ratio) # where to end fit on right of peak center
    
        
    # Get histogram of pulse times
    t_hist = np.histogram(t0,bins = nbins, range = (0,mod_length))
    x_vals = [(t_hist[1][i]+t_hist[1][i-1])/2 for i in range(1,len(t_hist[1]))] # Get the bin centers for the histogram
    peaks = find_peaks(t_hist[0],height=np.max(t_hist[0])/10) # Get peaks
    peak_max = max(peaks[1]['peak_heights'])
    #print(peak_max)
    
    peak_max_idx = np.argmax(peaks[1]['peak_heights'])
    peak_center = peaks[0][peak_max_idx]
        
    small_xvals = [] # Only need to fit around largest peak center
    
    
    small_xvals = x_vals[peak_center-left_off:peak_center+right_off]
    small_yvals = t_hist[0][peak_center-left_off:peak_center+right_off]
    
    mean_est = peak_center/ratio #Get an estimate for the mean of the gaussian
    parameters, covariance = curve_fit(Gauss, small_xvals, small_yvals, maxfev = 15000, p0 = [peak_max,mean_est,0.1])
    
    return parameters
"""    
    
    
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


def exponnorm_func2(t, t0, amp, tau, baseline):
    sigma = 0.96
    scale = sigma
    y0 = (t - t0) / scale
    y1 = (t - t0 + 1) / scale
    k = tau / sigma
    return 2048 + baseline - amp * 100 * (stats.exponnorm.cdf(y1, k, loc=0, scale=scale) -
                                          stats.exponnorm.cdf(y0, k, loc=0, scale=scale))
