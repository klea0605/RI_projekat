import numpy as np
import os

# Kad se preuzmu podaci kao sa OpenNeuro zvace se svi uniformno 

def sub_name(sub_number):
    # Standarize subjects name
    if sub_number < 10:
        sub_num_str = 'sub-0'+str(sub_number)
    else:
        sub_num_str = 'sub-'+str(sub_number)
        
    return sub_num_str


# Ucitava .dat podatke (4 kolone: sample, trial class, trial condition, trial session)
def load_events(root_dir, subject_num, session):
    
    subject_str = sub_name(subject_num)

    file_name = root_dir + "/derivatives/" + subject_str + "/ses-0" + str(session) + "/" + subject_str + "_ses-0" + str(session) + "_events.dat"
    events = np.load(file_name, allow_pickle = True) # recimo da verujemo Nietu
    
    return events


def save_results_for_participant(module, model_name, results_save, save_dir, Condition, subject, accuracy_list):
     if results_save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file_name = save_dir + "_" + module + "_" + model_name + "_AVG_CV_" + Condition  + "_Subject_" + str(subject)+ ".npy"

        np.save(file_name, accuracy_list)



# Iz prosledjenih X i Y izdvajam samo one redove za koje je ispunjen inner speech condition (posto to jedino klasifikujem)
# Evo recimo ovo sam sama napisala
def Filter_by_condition(X, Y, condition):
    condition = condition.upper()

    if condition == 'ALL':
        return X, Y
    
    elif condition == "IN" or condition == "INNER":
        inner_ind = 1
        indices = [i for i in range(Y.shape[0]) if Y[i, 2] == inner_ind]
        X_n = X[indices]
        Y_n = Y[indices]

    else:
        raise Exception("Treba da bude inner speech recognition")

    return X_n, Y_n

def Filter_by_class(X, Y, Class):
    Class = Class.upper()

    if Class == "ALL":
        return X, Y
    
    X_n = []
    Y_n = []
    indicator = 0 # podrazumevano UP
    
    if Class == "DOWN":
        indicator = 1
    elif Class == "RIGHT":
        indicator = 2
    elif Class == "LEFT":
        indicator = 3

    indices = [i for i in range(Y.shape[0]) if Y[i][1] == indicator]
    X_n = X[indices]
    Y_n = Y[indices]

def Transform_for_classificator(X, Y, Conditions, Classes):   # Npr: Conditions = ["In", "Pron"]  Classes = [["left", "right"], ["All"]]
    num_conditions = len(Conditions)
    num_classes = len(Classes)
    
    X_final = []
    Y_final = []

    if num_conditions != num_classes:
        raise Exception("Za svaki uslov mora da se prosledi lista klasa")

    for i in range(num_conditions):
        condition = Conditions[i]
        classes_list = Classes[i]

        X_conditioned, Y_conditioned = Filter_by_condition(X, Y, condition)

        for j in range(len(classes_list)):
            curr_class = classes_list[j]
            X_filtered, Y_filtered = Filter_by_class(X_conditioned, Y_conditioned, curr_class)

            if i == 0 and j == 0:
                X_final = X_filtered
                Y_final = Y_filtered
            else:
                X_final = np.vstack([X_final, X_filtered])
                Y_final = np.vstack([Y_final, Y_filtered])

    return X_final, Y_final[:, 1] 
