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
        'implant_rot': -64.74,
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
        'implant_x': -1896,
        'implant_y': -542,
        'implant_rot': -44,
        'loc_od_x': 15.8,
        'loc_od_y': 1.4,
        'xmin': -30,
        'xmax': 30,
        'ymin': -22.5,
        'ymax': 22.5
    },
    '51-009': {
        'subject_id': 'S3',
        'implant_type_str': 'ArgusII',
        'implant_x': -1203,
        'implant_y': 280,
        'implant_rot': -35,
        'loc_od_x': 15.4,
        'loc_od_y': 1.57,
        'xmin': -32.5,
        'xmax': 32.5,
        'ymin': -24.4,
        'ymax': 24.4
    },
    '52-001': {
        'subject_id': 'S4',
        'implant_type_str': 'ArgusII',
        'implant_x': -1945,
        'implant_y': 469,
        'implant_rot': -34,
        'loc_od_x': 15.8,
        'loc_od_y': 1.51,
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


def average_trials(df):
    """
    Takes in a dataframe, and contains logic for aggregating the different fields across trials. 
    For example, some fields like 'image' and 'num_regions' should just be averaged, some fields
    like 'amp' and 'freq' should be constant, some fields like 'filename' no longer make sense,
    and some fields such as regionprops measurements should (maybe) be recalculated on the averaged images
    """
    # TODO
    pass


def save_shapes(df, h5_file, ignore_overwrite=False):
    """
    Saves df to h5 file.
    Note that the h5 file will be completely overwritten to the new dataframe,
    so make sure that this new dataframe still contains all of the data you wish to save
    """
    if not ignore_overwrite and len([s for s in subject_params.keys() if s not in df['subject'].unique()]) > 0:
        raise ValueError("Missing subjects in new dataframe, use with ignore_overwrite=True to force save")

    stim_types = ['CDL0.35', 'CDL0.75', 'MultiElectrode', 'SingleElectrode', 'SpatialSummation',
                  'Step1', 'Step1a', 'Step1b', 'Step1c', 'Step1d', 'Step2&3', 'Step2a', 'Step3',
                  'Step4', 'Step5', 'Step6a']
    if not ignore_overwrite and len([s for s in stim_types if s not in df['stim_class'].unique()]) > 0:
        raise ValueError("Missing stim_class in new dataframe, use with ignore_overwrite=True to force save")

    
    """Converts the data from Pandas DataFrame to HDF5"""
    Xy = df.copy()

    Xy['image'] = pd.Series([row.image for (_, row) in Xy.iterrows()], index=Xy.index)

    # Convert images into black and white: (already black and white)
    Xy.image = Xy.image.apply(lambda x: x.astype(np.bool))
    
    # Split 'img_shape' into 'img_shape_x' and 'img_shape_y':
    Xy['img_shape_x'] = Xy.img_shape.apply(lambda x: x[0])
    Xy['img_shape_y'] = Xy.img_shape.apply(lambda x: x[1])
    Xy.drop(columns='img_shape', inplace=True)
    
        
    file = h5py.File(h5_file, 'w')
    # split by subject so each can be loaded individually
    for subject, data in Xy.groupby('subject'):
        # Image data:
        file.create_dataset("%s.image" % subject,
                            data=np.array([row.image
                                           for (_, row) in data.iterrows()]),
                           dtype=np.bool_)
        # String data:
        for col in ['subject', 'filename', 'stim_class', 'electrode1', 'electrode2', 'implant', 'date']:
            dt = h5py.string_dtype(encoding='utf-8')
            file.create_dataset("%s.%s" % (subject, col),
                                data=np.array([row[col]
                                               for (_, row) in data.iterrows()],
                                              dtype=dt))
        # Int data:
        for col in ['area', 'img_shape_x', 'img_shape_y', 'experiment', 'trial', 'num_regions']:
            file.create_dataset("%s.%s" % (subject, col),
                                data=np.array([row[col]
                                               for (_, row) in data.iterrows()],
                                              dtype=np.int32))
        # Float data:
        for col in ['amp1', 'amp2', 'freq', 'pdur', 'x_center', 'y_center', 'orientation',
                    'eccentricity', 'compactness', 'elec_delay']:
            file.create_dataset("%s.%s" % (subject, col),
                                data=np.array([row[col]
                                               for (_, row) in data.iterrows()],
                                              dtype=np.float32))
    file.close()