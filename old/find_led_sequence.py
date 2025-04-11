import numpy as np
import matplotlib.pyplot as plt
import itertools
from pandas import *
import json


def find_leds(file_path, num_mpmts, n_coverage, diffusers_only=True):
    
    """
    This function takes in a path to a csv file that contains the PMTs that are hit by each LED in the WCTE, the number of 
    mPMT-LED combinations one is wanting to use, and the number of times one desires each PMT to see light (n_coverage). 
    The function returns an appropriate list of mPMT-LED combinations to fire.
    
    NOTE: this code is wildly inefficient and could be written in a much more sensible way

    Note also that the current csv file labels the diffusers as '60 deg', but in reality they were simulated with an opening angle of 120 degrees
    
    
    file_path: type = string -- The path to the csv file
    num_mpmts: type = int -- the number of mPMT-LED combinations one is wanting to use
    n_coverage: type = int -- the minimum number of LEDs that are required to hit each PMT
    diffusers_only: type = bool -- allows one to select only LEDs that are diffusers (should increase efficiency)
    
    By Jakob Rimmer
    
    """
    
    # reading CSV file
    data = read_csv(file_path)
    
    # converting column data to list
    slot_id_led = data['Slot_id'].tolist()
    led_pos = data['LED_Pos_id'].tolist()
    led_type = data['led_col_type'].tolist()
    led_diff_idx = np.where(np.array(led_type)=='60 deg')[0]
    pmt_raw = data['Unique IDs of PMTs that received light'].tolist()
    pmts = []
    for i in range(len(slot_id_led)):
        pmts.append(json.loads(pmt_raw[i]))
        
        
        
    # Try randomly selecting a given number of mPMTs and LEDs and see if we can find a combination
    no_led = [95,31,35,26,19,57,5,27,32,45,74,77,79,85,91,99,9] #mPMTs that should not be used
    
    break_loop = False
    pmts_not_hit = True
    
    # First, loop through and find a combination of LEDs that cover the entire WCTE
    for k in range(1000):
        unique = []
        pmts_guess = []
        mpmt_led_guess = []
        
        # Randomly select mPMT numbers
        for i in range(num_mpmts):
            mpmt = np.random.randint(low=0,high=106)
            if np.isin(mpmt,no_led):
                continue
                
        #print(np.where(np.array(slot_id_led) == mpmt)[0])
            if len(np.where(np.array(slot_id_led) == mpmt)[0])==0:
                continue
                
            # Randomly select LEDs corresponding to the mPMT slot numbers
            rand = np.random.randint(low=0,high=3)
            idx = np.where(np.array(slot_id_led) == mpmt)[0][rand]
            
            # If we only want to use diffusers, then only consider diffuser indices
            if diffusers_only == True:
                try:
                    idx = np.intersect1d(np.where(np.array(slot_id_led) == mpmt)[0], led_diff_idx)[0]
                except:
                    continue
                    
            led = led_pos[idx]
            for j in range(len(pmts[idx])):
                pmts_guess.append([mpmt, led, pmts[idx][j]])
            
            mpmt_led_guess.append([mpmt, led])
       
        
        pmts_guess = np.array(pmts_guess)
        mpmt_led_guess = np.array(mpmt_led_guess)
        
        # We want to leave room to add LEDs that hit PMTs which are not hit many times
        if len(mpmt_led_guess) > 0.8*num_mpmts:
            continue
        
        # If there are any repeated entries in the mPMT-LED array, then try again
        if max(np.unique(mpmt_led_guess,axis=0,return_counts=True)[1])>1:
            continue
     
        # Make sure all PMTs see light
        if len(np.unique(pmts_guess[:,2])) == 1767:
            
            unique = mpmt_led_guess
            pmts_not_hit = False
            
                
            break_loop=True
            break
            
            
        if break_loop:
            break

    if pmts_not_hit == True:
        return print('Could not find LEDs to accomplish full coverage of WCTE. Please try again.')
    
    # We now have a set of mPMTs and LEDs that hit all PMTs, but let's find the PMTs that are only
    # hit by one LED, then compile a list of mPMT-LED combinations that hit those PMTs.
    # We can then add some of these combinations to our list of LEDs in the hopes that we can
    # hit each PMT with at least n_coverage LEDs...
    low_hit_pmts = []
    mpmt_led_small_hit = [] # Array of mPMT slot id and LED positions that hit low-hit PMTs
    idx_mpmt_led_small_hit = []
    for i in range(len(np.unique(pmts_guess[:,2], return_counts=True)[1])):
    
        #Check to ensure we are picking PMTs with a low number of LEDs that hit them
        if np.unique(pmts_guess[:,2], return_counts=True)[1][i] < n_coverage:
            # cycle through PMT list as ordered by mPMT and LED number
            for j in range(len(pmts)):
                if np.isin(np.unique(pmts_guess[:,2], return_counts=True)[0][i], pmts[j]):
                    idx = j
                    
                    if diffusers_only == True:
                        if led_type[idx] != '60 deg':
                            continue
                        
                    if np.isin([slot_id_led[idx],led_pos[idx]],mpmt_led_small_hit).all():
                        continue
                    if np.isin([slot_id_led[idx],led_pos[idx]],unique).all():
                        continue
                    else:
                        mpmt_led_small_hit.append([slot_id_led[idx],led_pos[idx]])
                        idx_mpmt_led_small_hit.append(idx)
                        
    
    # For ease, put data into a single array format...should've done this previously...
    all_data = [slot_id_led,led_pos,pmts]
    all_data_nice = []
    mpmt_led_comb = []
    for i in range(len(all_data[0])):
        all_data_nice.append([all_data[0][i],all_data[1][i],all_data[2][i]])
        mpmt_led_comb.append([all_data[0][i],all_data[1][i]])
        
    mpmt_led_comb = np.array(mpmt_led_comb)
                        
                 
    # We now have a list of mPMTs and LEDs to choose from that will help to hit each PMT n_coverage times
    # Let's add some of these mPMT-LED combinations to our current list until each PMT in WCTE is hit by
    # at least n_coverage LEDs
    
    
    #Try cycling through 100 times, each time checking which PMTs only see light from one LED

    for i in range(100):
        extra_leds = []
        
        # If the list of "small hit" LEDs is not empty, then select mPMT-LED combinations from this list
        if len(mpmt_led_small_hit) != 0:
            rand_idx = np.random.randint(low=0,high=len(mpmt_led_small_hit),size = num_mpmts-len(unique))
            for j in rand_idx:
                extra_leds.append(mpmt_led_small_hit[j])
            
        # If there are no "low hit" PMTs then just sample from the remaining LEDs
        else:
            rand_idx = np.random.randint(low=0,high=len(mpmt_led_comb),size = num_mpmts-len(unique))
            for j in rand_idx:
                extra_leds.append(mpmt_led_comb[j])
        
        repeat = False
        
    
        # If we have repeated entries in mpmt-LED array then try again
        for k in range(len(extra_leds)):
            if (np.array(extra_leds[k]) == unique).all(axis=1).any():
                repeat =True
        if repeat == True:
            continue
        
        unique_app = np.concatenate((unique,np.array(extra_leds)))
        if max(np.unique(unique_app,axis=0,return_counts=True)[1]) >1:
            continue
        
        #Now test to see how many PMTs only see light from one LED
    
        test_pmts = []
        idx = []
        
        for k in range(len(unique_app)):
            #Get all mPMT slot indices 
            idx_mpmt = np.where(slot_id_led == unique_app[k][0])
            
            
            #Cycle through possible mPMT slot indices to get correct LED index
            for j in range(len(idx_mpmt[0])):
                if led_pos[int(idx_mpmt[0][j])] == unique_app[k][1]:
                    idx.append(idx_mpmt[0][j])
    
        for l in idx:
            for j in range(len(pmts[l])):
                test_pmts.append(pmts[l][j])
            
            
        count = 0
        for p in range(len(np.unique(test_pmts, return_counts=True)[1])):
    
            if np.unique(test_pmts, return_counts=True)[1][p] < n_coverage:
                count+=1
        #print(count)
        if count <1:
            break
    if count > 0:
        return print('Count not find suitable sequence. Please run again or increase the number of mPMTs/LED combinations.')
            
    else: 
        return np.unique(unique_app,axis=0)
