import numpy as np
import pandas as pd
import h5py


try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False

try:
    import h5py
    has_h5py = True
except ImportError:
    has_h5py = False

# Dictionary of subject parameters
# This is unlikely to change often so its fine to have coded instead of in data
subject_params = {
    'TB': {
        'subject_id': 'S1',
        'implant_type_str': 'ArgusI',
        'implant_x': -1527,
        'implant_y': -556,
        'implant_rot': -1.13,
        'loc_od_x': 13.6,
        'loc_od_y': 0.0,
        'xmin': -36.9,
        'xmax': 36.9,
        'ymin': -36.9,
        'ymax': 36.9
    },
    '12-005': {
        'subject_id': 'S2',
        'implant_type_str': 'ArgusII',
        'implant_x': -1761,
        'implant_y': -212,
        'implant_rot': -0.188,
        'loc_od_x': 15.4,
        'loc_od_y': 1.86,
        'xmin': -30,
        'xmax': 30,
        'ymin': -22.5,
        'ymax': 22.5
    },
    '51-009': {
        'subject_id': 'S3',
        'implant_type_str': 'ArgusII',
        'implant_x': -799,
        'implant_y': 93,
        'implant_rot': -1.09,
        'loc_od_x': 15.7,
        'loc_od_y': 0.75,
        'xmin': -32.5,
        'xmax': 32.5,
        'ymin': -24.4,
        'ymax': 24.4
    },
    '52-001': {
        'subject_id': 'S4',
        'implant_type_str': 'ArgusII',
        'implant_x': -1230,
        'implant_y': 415,
        'implant_rot': -0.457,
        'loc_od_x': 15.9,
        'loc_od_y': 1.96,
        'xmin': -32,
        'xmax': 32,
        'ymin': -24,
        'ymax': 24
    }
}

def _hdf2df(hdf_file, desired_subjects=None):
    """Converts the data from HDF5 to a Pandas DataFrame"""
    f = h5py.File(hdf_file, 'r')
    
    # Fields names are 'subject.field_name', so we split by '.'
    # to find the subject ID:
    subjects = np.unique([k.split('.')[0] for k in f.keys()])
    if desired_subjects is not None:
        subjects = [s for s in subjects if s in desired_subjects]

    # Create a DataFrame for every subject, then concatenate:
    dfs = []
    for subject in subjects:
        df = pd.DataFrame()
        df['subject'] = subject
        for key in f.keys():
            if subject not in key:
                continue
            # Find the field name, that's the DataFrame column:
            col = key.split('.')[1]
            if col == 'image':
                # Images need special treatment:
                # - Direct assign confuses Pandas, need a loop
                # - Convert back to float so scikit_image can handle it
                df['image'] = [img.astype(np.float64) for img in f[key]]
            else:
                df[col] = f[key]
                # It seems that H5py behaves differently on different systems, sometimes this comes out as a string, sometimes bytes
                if col in ['subject', 'filename', 'stim_class', 'electrode1', 'electrode2', 'implant', 'date'] and type(df[col].iloc[0]) == bytes:
                    # convert from bytes to string
                    df[col] = df[col].apply(lambda x: x.decode('utf-8'))
        dfs.append(df)
    dfs = pd.concat(dfs)
    f.close()
    
    # Combine 'img_shape_x' and 'img_shape_y' back into 'img_shape' tuple
    dfs['img_shape'] = dfs.apply(lambda x: (x['img_shape_x'], x['img_shape_y']), axis=1)
    dfs.drop(columns=['img_shape_x', 'img_shape_y'], inplace=True)
    return dfs


def load_shapes(h5file, subjects=None, stim_class=['SingleElectrode', 'MultiElectrode'], implant=None,
                experimentID=None, shuffle=False, random_state=42):
    """
    Loads shapes from h5 file created from data-warehouse script

    Parameters
    ----------
    h5file : str
        Path to h5 file storing the data
    subjects : str | list of strings | None, optional
        Select data from a subject or list of subjects. By default, all
        subjects are selected. Subject can be either Second Sight ID (e.g. 12-005)
        or subject number (e.g. S2).
    stim_class : str | list of strings | None, optional
        Select data by stim_class. Options are 'MultiElectrode', 'SingleElectrode',
        'SpatialSummation', 'Step' + desired_step, 'CDL0.35', 'CDL0.75'. Default is 
        Single and Multi Electrode
    implant : 'ArgusI' | 'ArgusII' | None, optional
        Select data from either ArgusI or ArgusII implants. None will select
        both. You cannot select both by subject and by implant.
    experimentID : int | list of int | None, optional
        Select all trials with specific experiment IDs. None selects all 
        experiments
    shuffle : boolean, optional
        If true, shuffle the data
    """
    if not has_h5py:
        raise ImportError("You do not have h5py installed. "
                          "You can install it via $ pip install h5py.")
    if not has_pandas:
        raise ImportError("You do not have pandas installed. "
                          "You can install it via $ pip install pandas.")    

    if implant is not None and subjects is not None:
        raise ValueError("Select either by subject or by implant, not both")

    if implant is not None:
        if implant not in ["ArgusI", "ArgusII"]:
            raise ValueError("Implant must be one of 'ArgusI', 'ArgusII'")
        subjects = [k for k in subject_params.keys() if subject_params[k]['implant_type_str'] == implant]
    elif subjects is not None:
        if type(subjects) == str:
            subjects = [subjects]
        # Switch subject number to second sight ids
        alias = {subject_params[ssid]['subject_id']:ssid for ssid in subject_params.keys()}
        subjects = [alias[s] if s in alias.keys() else s for s in subjects]
        for s in subjects:
            if s not in subject_params.keys():
                raise ValueError("Unknown subject: " + s)
        
    df = _hdf2df(h5file, desired_subjects=subjects)

    if stim_class is not None:
        stim_classes = ['CDL0.35', 'CDL0.75', 'MultiElectrode', 'SingleElectrode', 'SpatialSummation',
                        'Step1', 'Step1a', 'Step1b', 'Step1c', 'Step1d', 'Step2&3', 'Step2a', 'Step3',
                        'Step4', 'Step5', 'Step6a']
        if type(stim_class) == str:
            stim_class = [stim_class]
        for stim_c in stim_class:
            if stim_c not in stim_classes:
                raise ValueError("Unknown stim_class: " + stim_c)
        df = df[df['stim_class'].isin(stim_class)]
    
    if experimentID is not None:
        if type(experimentID) == str:
            experimentID = [experimentID]
        # dont error here if one in list isn't present in df
        df = df[df['experiment'].isin(experimentID)]

    if shuffle:
        df = df.sample(n=len(df), random_state=random_state)
    return df.reset_index(drop=True)
