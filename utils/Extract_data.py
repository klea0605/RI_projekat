#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import mne
import gc


# In[3]:
from TOL_dataset_utils import sub_name, load_events
# In[4]:


# Ucitavanje eeg podataka za jednok coveka kroz sve 3 sesije (kolko glupa rec)

# Vraca subject_data_over_all_sessions, events  as np array 

def Extract_data_from_subject(root_dir, subject_number, datatype):

    data = dict()
    events = dict()
    Session_numbers = [1,2,3] 
    datatype = datatype.lower()
    
    for session in Session_numbers:

        subject_str = sub_name(subject_number)    
        events[session] = load_events(root_dir, subject_number, session)  # uzima iz .dat za ovog ucesnika
        
        if datatype=="eeg":
            #  load data and events
            file_name = root_dir + '/derivatives/' + subject_str + '/ses-0'+ str(session) + '/' + subject_str +'_ses-0'+str(session)+'_eeg-epo.fif'
            X= mne.read_epochs(file_name,verbose='WARNING')
            data[session]= X._data

        else:
            raise Exception("Invalid Datatype")
        

    X = np.vstack((data.get(1),data.get(2),data.get(3)))  # redja ih po kolonama znaci pristup je session-1
    Y = np.vstack((events.get(1), events.get(2), events.get(3))) 

    return X, Y


# In[6]:


"""
Prima listu ucesnika i ucitava sve sesije za svakog od njih i sve zajedno vraca kao X.
Sluzi da se model trenira i na ovakvim podacima (bolja generalizacija)

"""

def Extract_data_multisubject(root_dir, Subject_numbers, datatype):

    Session_numbers = [1, 2, 3]
    datatype = datatype.lower()

    # privremeno skladistenje dok ne ucitam sve ucesnike
    tmp_list_X = []
    tmp_list_Y = []

    # rekonstrukcija dimenzija za np array 
    rows = []
    chann = 0 
    steps = 0 
    columns = 0  

    total_elem = len(Subject_numbers)*3  # 3 sessions per subject
    i = 0

    for subject_number in Subject_numbers:       
        
        for session in Session_numbers:

            subject_number_str = sub_name(subject_number) 

            # base_file_name = root_dir + '/derivatives/' + subject_number_str + '/ses-0'+ str(session) + '/' + subject_number_str + '_ses-0'+str(session)
            events_file_name = base_file_name + '_events.dat'
            # data_tmp_Y = np.load(events_file_name,allow_pickle=True)  # ? sta radi allow pickle ?
            data_tmp_Y = load_events(root_dir, subject_number, session)
            tmp_list_Y.append(data_tmp_Y)

            if datatype=="eeg":
                # load data
                base_file_name = root_dir + '/derivatives/' + subject_number_str + '/ses-0'+ str(session) + '/' + subject_number_str + '_ses-0' + str(session)
                eeg_file_name = base_file_name+'_eeg-epo.fif'
                # print(f'Reading eeg file: {eeg_file_name}')
                data_tmp_X = mne.read_epochs(eeg_file_name, verbose='WARNING')._data
                # print(f'Num rows for session {session}: {data_tmp_X.shape}')
                # print(f'Num rows for session {session}: {len(data_tmp_Y)}')
                rows.append(data_tmp_X.shape[0])
                tmp_list_X.append(data_tmp_X)

                if i == 0 and session == 1: # assume same number of channels, time steps, and column labels in every subject and session; 
                  chann = data_tmp_X.shape[1] 
                  steps = data_tmp_X.shape[2]
                  columns = data_tmp_Y.shape[1]
        i += 1
      # Kraj ucitavanja podataka za sve prosledjene ucesnike.
 
    X = np.empty((sum(rows), chann, steps))     # np.empty sluzi da brze napravi np.array zeljene dimenzije;  Ne koristim tmp_list_X jer mi treba np arr 
    Y = np.empty((sum(rows), columns))          
    Y = Y.astype(int)  # empty mi stavlja float-ove; evo recimo ovo sam sama dodala
    offset = 0

    # smestanje podataka u niz
    # j je indeks nekog ucesnika (ali ne i njegov broj u eksperimentu)
    for j in range(total_elem):

      X[offset : offset+rows[j], : , :] = tmp_list_X[0]   # ovo samo dodaje sve redove j-tog ucitanog u np.array; ! Uvek koristi tmp_list_X[0] jer u svakoj iteraciji brise !
      
      # events are available only for eeg and exg datatypes 
      if datatype == 'eeg' or datatype == "exg":
        Y[offset : offset+rows[j], :] = tmp_list_Y[0] 
        
      offset += rows[j]
      del tmp_list_X[0]  # !! brise zbog memorije
      del tmp_list_Y[0]  
      gc.collect() 

    # Kraj formatiranja ucitanih ucesnika u np

    if datatype == "eeg" or datatype=="exg":
      return X,Y    
  
    else:
       return X


# In[ ]:




