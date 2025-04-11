# This file provides a function from Laurence that takes in the resulting dictionary of process_parquet and separates it into lists of dictionaries according to which LED is firing
import time
import pickle
import numpy as np

def separate_dict(filepath,dict_files):

    #open the first file and get a list 
    #filepath = "/eos/experiment/wcte/data/2024_beam_root_data/20241126133507_interspill_LED_"+str(5)+".dict"  # Update this with your actual path
    # Open and load the pickle file
    
    led_mpmts = {}
    sorted_dicts = []
    
    for idict in dict_files:
        file = filepath+idict  
        # Open and load the pickle file
        with open(file, 'rb') as f:
            pq_data = pickle.load(f)
    
        #unique_card_ids = list(set(item['card_id'] for item in pq_data))
        cards = []
        for i in range(len(pq_data)):
            cards.append(pq_data[i]['card_id'])
        unique_card_ids = np.unique(cards)
        
        print('Card IDs for file ' + str(idict) + ': ' + str(unique_card_ids))
                
        for i in range(len(pq_data)):
            if str(pq_data[i]['card_id']) not in led_mpmts:
                led_mpmts[str(pq_data[i]['card_id'])] = []
                
            led_mpmts[str(pq_data[i]['card_id'])].append(pq_data[i])
     
        del pq_data
        
    for card in led_mpmts:
        sorted_dicts.append(led_mpmts[card])
        
    return sorted_dicts

            
    
''' 
    
    with open(filepath+dict_files[0], 'rb') as f:
        pq_data = pickle.load(f)
    print(pq_data[0].keys())
    unique_card_ids = list(set(item['card_id'] for item in pq_data))
    print(unique_card_ids)
    del pq_data
 
    sorted_dicts = []
    
    for led_mPMT in unique_card_ids:
        print("Process mPMT",led_mPMT)    
        single_led_list = []
        for iDict in dict_files:
            start = time.time()
            file = filepath+iDict  
            # Open and load the pickle file
            with open(file, 'rb') as f:
                pq_data = pickle.load(f)
                for data in pq_data:
                    if(data['card_id']==led_mPMT):
                        single_led_list.append(data) 
            end = time.time()
            #print("time",end-start,"list len",len(single_led_list))
        #print(len(single_led_list))
        sorted_dicts.append(single_led_list)
    
    return sorted_dicts
    
'''