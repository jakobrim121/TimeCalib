import numpy as np
import matplotlib.pyplot as plt
import itertools
from pandas import *
import json


def find_leds(file_path, num_mpmts, n_coverage):
    
    """
    This function takes in a path to a csv file that contains the PMTs that are hit by each LED in the WCTE, the number of 
    mPMT-LED combinations one is wanting to use, and the number of times one desires each PMT to see light (n_coverage). 
    The function returns an appropriate list of mPMT-LED combinations to fire.
    
    NOTE: this code is wildly inefficient and could be written in a much more sensible way
    
    
    file_path: type = string -- The path to the csv file
    num_mpmts: type = int -- the number of mPMT-LED combinations one is wanting to use
    n_coverage: type = int -- the minimum number of LEDs that are required to hit each PMT
    
    By Jakob Rimmer
    
    """
    
    # reading CSV file
    data = read_csv(file_path)
    
    # converting column data to list
    slot_id_led = data['Slot_id'].tolist()
    led_pos = data['LED_Pos_id'].tolist()
    pmt_raw = data['Unique IDs of PMTs that received light'].tolist()
    pmts = []
    for i in range(len(slot_id_led)):
        pmts.append(json.loads(pmt_raw[i]))
        
        
        
    # Try randomly selecting a given number of mPMTs and LEDs and see if we can find a combination
    no_led = [95,31,35,26,19,57,5,27,32,45,74,77,79,85,91,99,9] #mPMTs that should not be used
    barrel_mpmts = range(21,85)
    bec_mpmts = range(0,21)
    tec_mpmts = range(85,106)
    
    barrel_mpmts_sel = []
    bec_mpmts_sel = []
    tec_mpmts_sel = []
    
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
            led = led_pos[idx]
            for j in range(len(pmts[idx])):
                pmts_guess.append([mpmt, led, pmts[idx][j]])
            
            mpmt_led_guess.append([mpmt, led])
            
            
        
        
        pmts_guess = np.array(pmts_guess)
        mpmt_led_guess = np.array(mpmt_led_guess)
        
        # If there are any repeated entries in the mPMT-LED array, then try again
        if max(np.unique(mpmt_led_guess,axis=0,return_counts=True)[1])>1:
            continue
     
        # Make sure all PMTs see light
        if len(np.unique(pmts_guess[:,2])) == 1767:
            
            unique = mpmt_led_guess
            pmts_not_hit = False
            
            for m in range(len(unique)):
                if np.isin(unique[m,0],barrel_mpmts):
                    barrel_mpmts_sel.append(unique[m])
                elif np.isin(unique[m,0],bec_mpmts):
                    bec_mpmts_sel.append(unique[m])
                elif np.isin(unique[m,0],tec_mpmts):
                    tec_mpmts_sel.append(unique[m])
          
                
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
                        
    
    
    # We'll probably want to fire some LEDs together to minimize the time taken for calibration via LEDs,
    # so try grouping the LEDs in pairs (make sure barrel mPMTs fire together)
    # ***This idea is a work in progress and is not currently what this function returns***
    
    pairs = {}
    pairs['barrel'] = []
    pairs['tec'] = []
    pairs['bec'] = []
    
    idx_barrel = []
    for i in range(len(barrel_mpmts_sel)):
        for j in range(len(mpmt_led_comb)):
            if (barrel_mpmts_sel[i] == mpmt_led_comb[j]).all():
                idx_barrel.append(j) #Get index of mPMT-LED combination in original data
                j = len(mpmt_led_comb)
    
    idx_bec = []
    for i in range(len(bec_mpmts_sel)):
        for j in range(len(mpmt_led_comb)):
            if (bec_mpmts_sel[i] == mpmt_led_comb[j]).all():
                idx_bec.append(j) #Get index of mPMT-LED combination in original data
                j = len(mpmt_led_comb)
                
    idx_tec = []
    for i in range(len(tec_mpmts_sel)):
        for j in range(len(mpmt_led_comb)):
            if (tec_mpmts_sel[i] == mpmt_led_comb[j]).all():
                idx_tec.append(j) #Get index of mPMT-LED combination in original data
                j = len(mpmt_led_comb)


    indices_paired_barrel = []

    for index1 in idx_barrel:
        no_overlap_sub = []
        for index2 in idx_barrel:
            if len(np.intersect1d(pmts[index1],pmts[index2])) == 0:
                if len(np.intersect1d(index1,indices_paired_barrel))==0 and len(np.intersect1d(index2,indices_paired_barrel))==0:
                    pairs['barrel'].append([mpmt_led_comb[index1], mpmt_led_comb[index2]])
                    indices_paired_barrel.append(index1)
                    indices_paired_barrel.append(index2)
                    
    unused_idx_barrel = np.setdiff1d(idx_barrel,indices_paired_barrel)
    
    
    
    #print(unused_idx_barrel,indices_paired_barrel,idx_barrel)
                        
    # We now have a list of mPMTs and LEDs to choose from that will help to hit each PMT n_coverage times
    # Let's add some of these mPMT-LED combinations to our current list until each PMT in WCTE is hit by
    # at least n_coverage LEDs
    
    # First find which mPMTs are in the barrel in this smaller set of mPMTs
    mpmt_led_small_hit_barrel = []
    mpmt_led_small_hit_tb = []
    for small_idx in range(len(mpmt_led_small_hit)):
        if np.isin(mpmt_led_small_hit[small_idx][0],barrel_mpmts):
            mpmt_led_small_hit_barrel.append(mpmt_led_small_hit[small_idx])
        else:
            mpmt_led_small_hit_tb.append(mpmt_led_small_hit[small_idx])
    
    mpmt_led_small_hit_tb = np.array(mpmt_led_small_hit_tb)
    mpmt_led_small_hit_barrel = np.array(mpmt_led_small_hit_barrel)
    
                
    
    
    #Try cycling through 100 times, each time checking which PMTs only see light from one LED
    barrel_filled = False
    barrel_mpmts_left = len(unused_idx_barrel)
    for i in range(100):
        extra_leds = []
        
        
        #rand_idx_barrel = np.random.randint(low=0,high=len(mpmt_led_small_hit_barrel),size = barrel_mpmts_left)   
        #rand_idx_tb = np.random.randint(low=0,high=len(mpmt_led_small_hit_tb),size = num_mpmts-len(unique)-barrel_mpmts_left)
        
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
        extra_leds_barrel = []
        extra_leds_tb = []
        
        
        
    
        #for j in rand_idx_barrel:
         #   extra_leds.append(mpmt_led_small_hit_barrel[j])
          #  extra_leds_barrel.append(mpmt_led_small_hit_barrel[j])
            
       # for j in rand_idx_tb:
        #    extra_leds.append(mpmt_led_small_hit_tb[j])
         #   extra_leds_tb.append(mpmt_led_small_hit_tb[j])
        
    
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