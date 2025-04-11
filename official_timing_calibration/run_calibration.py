# This file is used to run the timing calibration, save any pre-calibration data, and save any "bad" paths

import sys
import os.path
sys.path.insert(0, "../../Geometry")
sys.path.insert(0, "../../TimeCal")
sys.path.insert(0, "../../TimeCal/TimeCal")
from format_dictionary import separate_dict
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from Geometry.WCD import WCD
from calibration_pipeline_official import *
from TC_Simulator import *
from TC_Data import *
from TC_Calibrator import *
from TC_Multilaterator import *

plt.ioff()



### CONFIGURE YOUR SETTINGS BEFORE RUNNING ###

# Below are the file paths you need to specify

tooldaq = False
standalone = True

dict_folder = '/eos/user/j/jrimmer/led_data/dictionaries/tooldaq/self_trigger/1453/' # Dictionary folder
dict_folder = '/eos/user/j/jrimmer/led_data/dictionaries/2025-04-04/'


bad_path_save_folder = "/eos/user/j/jrimmer/SWAN_projects/official_led_calibration/bad_paths/" # Folder for saving bad paths

precal_save_folder = "/eos/user/j/jrimmer/led_data/precal/2025-04-04/" # Where the pre-calibration data is saved

# Where the calibration results are saved
calib_save_folder = "/eos/user/j/jrimmer/SWAN_projects/official_led_calibration/results/2025-04-04/" 


data_date = "2025_04_04" # The date the data was collected (format should be yyyy_mm_dd)

interactive = True # If running as a batch job, this should be false. Otherwise, you will be prompted for responses on command line

use_skip_paths = True # If you want to exlude the existing "bad" paths from previous calibrations


get_precal = False # Whether or not you want to calculate the pre-calibration data
run_calibration = True # Whether or not you want to actually run the calibration

DAC = 750 # LED intensity (hopefully this will be embedded in the LED files eventually)

# Whether or not you want to use all files in dictionary folder. If false, specify the start and end indices below
use_all_dict_files = False
start_file_num_dict = 16 # Starting file number in dictionary folder 
end_file_num_dict = 19 # Ending file number in dictionary folder

# Whether or not you want to use all files in precal folder. If false, specify the start and end indices below
use_all_precal_files = True
start_file_num_precal = 0 # Starting file number in precal folder (only need this if precal data is already calculated)
end_file_num_precal = 5 # Ending file number in precal folder (only need this if precal data is already calculated)

ref_mpmt_slot = 49 # Reference mPMT being used in calibration


### END OF CONFIGURATION ###

if not os.path.exists(dict_folder):
    Path(dict_folder).mkdir(parents=True, exist_ok=True)
    
if not os.path.exists(precal_save_folder):
    Path(precal_save_folder).mkdir(parents=True, exist_ok=True)
    
if not os.path.exists(calib_save_folder):
    Path(calib_save_folder).mkdir(parents=True, exist_ok=True)



n = 1.34 # Refractive index of water

tc_data = []
card_id_all = []
runs_unfiltered = []
runs_all = []

if use_skip_paths == True:

    try:
        skip_paths = np.load(bad_path_save_folder+"bad_paths_all.npy")
    
    except:
        skip_paths = [[999,999,999,999]]
        pass
else:
    skip_paths = [[999,999,999,999]]

# If you're wanting to calculate the pre-calibration data...
 
if get_precal == True:
    
    if use_all_dict_files == True:
        num_files = len(os.listdir(dict_folder))
        runs_unfiltered = os.listdir(dict_folder)
    
    else:
        num_files = end_file_num_dict - start_file_num_dict 
        runs_unfiltered = os.listdir(dict_folder)[start_file_num_dict:end_file_num_dict]
        
    first_good_file = 0
   
    for i in range(len(runs_unfiltered)):
        
        run = runs_unfiltered[i]
       
        
        if run[-4:] != 'dict' and run[-3:] != 'pkl':
            num_files -= 1
            
            first_good_file +=1
            
            continue
            
        if run[-8:] == 'bad.dict':
            num_files -= 1
            
            first_good_file +=1
            
            continue
        
        runs_all.append(run)
            
        print('Loading run: ' + str(run))
        
        if standalone == True:
            if i == first_good_file:
            
                start_run_num = run[run.find(data_date[:4]) : run.find(data_date[:4]) + 14]
                end_run_num = run[run.find(data_date[:4]) : run.find(data_date[:4]) + 14]
            
        
            if run[run.find(data_date[:4]) : run.find(data_date[:4]) + 14] < start_run_num:
                start_run_num = run[run.find(data_date[:4]) : run.find(data_date[:4]) + 14]
            
            if run[run.find(data_date[:4]) : run.find(data_date[:4]) + 14] > end_run_num:
                end_run_num = run[run.find(data_date[:4]) : run.find(data_date[:4]) + 14]
                
        else:
            start_run_num = run[10:14]
            end_run_num = run[10:14]
            
    # Use Laurence's function to sort the dictionaries as needed
    print('Sorting dictionaries...')
    sorted_dict = separate_dict(dict_folder,runs_all)
    
    # Loop through all dictionary files
    for i in range(len(sorted_dict)):
      
        
        data_all = sorted_dict[i]
        
        # LED number 4 does not exist, so label it LED number 3 if it occurs in the dictionary    
        card_id = data_all[0]['card_id']
        card_id_all.append(card_id)
        led_no = data_all[0]['led_no']
        if led_no == 4:
            data_all[0]['led_no'] = 3
          
      
        tc_data.append(get_precal_data(data_all,skip_paths))
        print('Number of PMTs hit by this LED after cuts: ' + str(len(tc_data[i])))
   
           
            
# If you're wanting to run the calibration on this data...

if run_calibration == True:
    from Geometry.Device import Device
    my_hall = Device.open_file('wcte_bldg157.geo')
    my_wcte = my_hall.wcds[0]
    tc_data_label = 0

    my_wcte.prop_design['refraction_index'] = n
    tc_led_data = TC_Data(str(tc_data_label))
    
    # If no pre-calibration data was calculated, then extract the data you're wanting to do calibration with...
    if get_precal == False:
        
        if use_all_precal_files == True:
            num_files = len(os.listdir(precal_save_folder))
            
            runs = os.listdir(precal_save_folder)
            card_id_all = runs
    
        else:
            num_files = end_file_num_precal - start_file_num_precal 
            runs = os.listdir(precal_save_folder)[start_file_num_precal:end_file_num_precal]
            card_id_all = runs
         
       
        # Load in all the pre-saved precal data
        for i in range(len(runs)):
            
            if i == 0:
                start_run_num = runs[i][runs[i].find(data_date[:4]) : runs[i].find(data_date[:4]) + 14]
                end_run_num = runs[i][runs[i].find(data_date[:4]) : runs[i].find(data_date[:4]) + 14]
        
            if runs[i][runs[i].find(data_date[:4]) : runs[i].find(data_date[:4]) + 14] < start_run_num:
                start_run_num = runs[i][runs[i].find(data_date[:4]) : runs[i].find(data_date[:4]) + 14]
            
            if runs[i][runs[i].find(data_date[:4]) : runs[i].find(data_date[:4]) + 14] > end_run_num:
                end_run_num = runs[i][runs[i].find(data_date[:4]) : runs[i].find(data_date[:4]) + 14]
            
            
            tc_data.append(np.load(precal_save_folder+runs[i]))
    
    # Cycle through the precal data and perform calibration on it
    for i in range(len(tc_data)):
        for j in range(len(tc_data[i])):
            
            if use_skip_paths == True:
                if max(np.unique(np.concatenate((skip_paths,[[int(tc_data[i][j][0]),int(tc_data[i][j][1]),int(tc_data[i][j][2]),int(tc_data[i][j][3])]])),axis=0,return_counts=True)[1])>1:
                    
                    continue
                else:
                    tc_led_data.set(int(tc_data[i][j][0]),int(tc_data[i][j][1]),int(tc_data[i][j][2]),int(tc_data[i][j][3]),tc_data[i][j][4],tc_data[i][j][5])
                
            else:
                tc_led_data.set(int(tc_data[i][j][0]),int(tc_data[i][j][1]),int(tc_data[i][j][2]),int(tc_data[i][j][3]),tc_data[i][j][4],tc_data[i][j][5])
    
    # Set up calibration
    wcte_calib = TC_Calibrator(my_wcte) 
    wcte_calib.assign_data(tc_led_data)
    wcte_calib.set_reference_mpmt(ref_mpmt_slot,0,0.1) # Set the reference mPMT
    wcte_calib.set_priors(0.,100.,0.,100.,epsilon_apply=False, alpha_apply=True) #Set the priors

    chi2, n_dof, chisqs, devs_led, dists_led, bad_paths_calib =wcte_calib.calibrate(return_chisqs = True,place_info='est') # Do the calibration
    
    if interactive == True:
        print('Save calibration results? Answer yes or no')
        save_calib = input()
    else:
        save_calib= 'yes'
        
    if save_calib == 'yes':
 
        estimated_clock_offsets = []

        estimated_led_delays = []

        estimated_pmt_delays = []

        mpmt_slots = {}

            
        for i in range(len(my_wcte.mpmts)):
            mpmt_slots[str(i)] = {}
            mpmt_slots[str(i)]['leds'] = {}
            mpmt_slots[str(i)]['pmts'] = {}
    
            try:
                mpmt_slots[str(i)]['clock offset'] = my_wcte.mpmts[i].prop_est['clock_offset']
                estimated_clock_offsets.append(my_wcte.mpmts[i].prop_est['clock_offset'])
            except:
                continue
        
            for il, led in enumerate(my_wcte.mpmts[i].leds):
            
                if il <3:
                    try:
                        mpmt_slots[str(i)]['leds'][str(il)] = led.prop_est['delay']
                        estimated_led_delays.append(led.prop_est['delay'])
                    except:
                        continue
                    
            for j in range(len(my_wcte.mpmts[i].pmts)):
            
                try:
                    mpmt_slots[str(i)]['pmts'][str(j)] = my_wcte.mpmts[i].pmts[j].prop_est['delay']
                    estimated_pmt_delays.append(my_wcte.mpmts[i].pmts[j].prop_est['delay'])
                except:
                    continue

            
        devs = []
        dists = []
        for key in devs_led:
            devs.append(devs_led[key])
            dists.append(dists_led[key])
            
            
        fig,ax = plt.subplots(1,4,figsize=(10,3))
        ax[0].hist(estimated_clock_offsets,bins=30)
        ax[1].hist(estimated_led_delays,bins=40)
        ax[2].hist(estimated_pmt_delays,bins=30)
        ax[3].hist(devs,bins=np.arange(-2,2,0.04))


        ax[0].set_title('mPMT Clock Offsets')
        ax[1].set_title('LED Delays')
        ax[2].set_title('PMT Delays')
        ax[3].set_title('Deviations')

        ax[0].set_xlabel('Time (ns)')
        ax[1].set_xlabel('Time (ns)')
        ax[2].set_xlabel('Time (ns)')
        ax[3].set_xlabel('Time (ns)')

        ax[3].text(0.5,0.9,'Mean: ' + str(round(np.mean(devs),4)),transform=ax[3].transAxes)
        ax[3].text(0.5,0.8,'STD: ' + str(round(np.std(devs),4)),transform=ax[3].transAxes)


        fig.tight_layout()
        
        if get_precal == False:
            num_files = end_file_num_precal-start_file_num_precal

        plt.savefig(calib_save_folder + "plots_"+str(len(card_id_all))+"LEDs_refSlot" + str(ref_mpmt_slot) + '_' + data_date + ".png")
        plt.close(fig)

        np.save(calib_save_folder + "devs_"+str(len(card_id_all))+"LEDs_refSlot" + str(ref_mpmt_slot) + '_'  + data_date + ".npy",devs)
    
        with open(calib_save_folder + "all_constants_"+str(len(card_id_all))+"LEDs_refSlot" + str(ref_mpmt_slot) + '_' + data_date + ".dict", 'wb') as fi:
            pickle.dump(mpmt_slots, fi)
            
        # Now save the dictionary that can be sent to the calibration database
        # Timing constants should be subtracted from raw arrival times (since here we have constant = -mPMT clock offset + PMT delay)
        
        single_constants = []
        
        for slot in mpmt_slots:
    
            for pmt in mpmt_slots[slot]['pmts']:
                try:
                    single_constants.append({'channel_id': int(slot)*100+int(pmt), 'timing_offset': -mpmt_slots[slot]['clock offset'] + mpmt_slots[slot]['pmts'][pmt]}) 
                    
                except:
                    pass
                
        
        start_time = start_run_num
        end_time = end_run_num
        official = False
        
        # Data to be written
        dictionary = {
        "start_run_number": start_run_num,
        "end_run_number": end_run_num,
        "start_time": start_run_num,
        "end_time": end_run_num,
        "calibration_name": "timing_offsets",
        "calibration_method": "LED",
        "official": official,
        "data": single_constants
        }
        
        # Serializing json
        json_object = json.dumps(dictionary, indent=4)
 
        # Writing to json file
        with open(calib_save_folder + "SR"+str(start_run_num)+"_ER" + str(end_run_num)+"_ST" + str(start_time)+"_ET"+str(end_time)+'.json', "w") as outfile:
            
            outfile.write(json_object)
            
            
    
    if len(bad_paths_calib) > 0 and interactive == True:
        print('Save list of bad paths? Answer yes or no.')
        answer1 = input()
        
    else:
        answer1 = 'no'
        
    if answer1 == 'yes':
        
        try:
            bad_list_on_file = list(np.load(bad_list_save_folder+'bad_paths_all.npy'))
            
        except:
            bad_list_on_file = [[999,999,999,999]]
        
        for i in range(len(bad_paths_calib)):
            if max(np.unique(np.concatenate((bad_list_on_file,[bad_paths_calib[i]])),axis=0,return_counts=True)[1])==1:
                bad_list_on_file.append(bad_paths_calib[i])
        
        np.save(bad_path_save_folder+'bad_paths_all.npy',bad_list_on_file)

if get_precal == True and interactive == True:        
    print('Save pre-calibration data? Answer yes or no')
    answer3 = input()
    
else:
    answer3 = 'no'
    
if answer3 == 'yes':
    
    if len(bad_paths_calib) > 0 and interactive == True:
        print('Save pre-calibration data with bad paths excluded? Answer yes or no')

        answer4 = input()
        
    else:
        answer4 = 'no'
    
    
    if answer4 == 'yes' and run_calibration == True:
        precal_data = []
    
        for i in range(len(tc_data)):
            precal_curr = []
            for j in range(len(tc_data[i])):
                if max(np.unique(np.concatenate((bad_paths_calib,[[tc_data[i][j][0],tc_data[i][j][1],tc_data[i][j][2],tc_data[i][j][3]]])),axis=0,return_counts=True)[1]) <2:
                    precal_curr.append(tc_data[i][j])
                
            precal_data.append(precal_curr)
                
         
      
    
        for i in range(len(precal_data)):
            fname = precal_save_folder+'card' + str(card_id_all[i])+'_'+str(start_run_num)
            
            if os.path.isfile(fname):
                existing_file = np.load(fname)
                if len(existing_file) > len(precal_data[i]):
                    continue
                    
            else:
            
                np.save(precal_save_folder+'card' + str(card_id_all[i])+'_'+str(start_run_num),precal_data[i])
                print('Saving file ' + precal_save_folder+'card' + str(card_id_all[i])+'_'+str(start_run_num) + '.npy')
            
            
    
            
        
    else:
        precal_data = []
    
        for i in range(len(tc_data)):
            
            precal_curr = []
            for j in range(len(tc_data[i])):
                
                precal_curr.append(tc_data[i][j])
                
            precal_data.append(precal_curr)
            
    
        for i in range(len(precal_data)):
            fname = precal_save_folder+'card' + str(card_id_all[i])+'_'+str(start_run_num)
            
            if os.path.isfile(fname):
                existing_file = np.load(fname)
                if len(existing_file) > len(precal_data[i]):
                    continue
                    
            else:
            
                np.save(precal_save_folder+'card' + str(card_id_all[i])+'_'+str(start_run_num),precal_data[i])
                print('Saving file ' + precal_save_folder+'card' + str(card_id_all[i])+'_'+str(start_run_num) + '.npy')
