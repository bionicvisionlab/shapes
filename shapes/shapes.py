import numpy as np
import pandas as pd
import h5py
import skimage.measure as measure 
from skimage.measure import label, regionprops

from pulse2percept.utils import center_image, shift_image

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
# Updated with new fits June 2021
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


def average_images(images, do_thresh=False, thresh=0.5, return_list=False):
    """
    Simple image averaging for binary images 
    - If do_thresh is false, each pixel in the returned image is 
      the mean of the corresponding pixel in each supplied image
    - If do_thresh is true, each pixel in the returned image will 
      be 1 if more than thresh percent of the corresponding pixels
      in the supplied images are 1
    """
    if do_thresh:
        img = (np.mean(images.apply(center_image), axis=0) > thresh).astype('float64')
    else:
        img = (np.mean(images.apply(center_image), axis=0)).astype('float64')
    if return_list:
        # this is mainly for pandas aggregration, which doesnt support ndarrays
        return list(img)
    return img

def average_trials(df, groupby=['implant', 'subject', 'amp1', 'amp2', 'electrode1', 'electrode2', 'freq', 'pdur', 'stim_class'], 
                   do_thresh=True, thresh=0.5):
    """
    Takes in a dataframe, and contains logic for aggregating the different fields across trials. 
    For example, some fields like 'image' and should be averaged, some fields
    like 'amp' and 'freq' should be constant, and some fields like 'filename' no longer make sense,
    """
    if not do_thresh:
        thresh = 0.0
    avg_df = df.reset_index().groupby(groupby).agg({'index' : 'unique', 'image' : lambda x: average_images(x, do_thresh=do_thresh, thresh=thresh, return_list=True)}).reset_index()
    # convert images back to np arrays
    avg_df['image'] = [np.array(x) for x in avg_df['image']]
    avg_df['num_regions'] = [len(measure.regionprops(measure.label(x > thresh))) for x in avg_df['image']]

    return avg_df

""" 
TODO:   This could be majorly sped up (don't worry about for now), 
should also be grouping by pulse_dur, stim_class, amp2, electrode2, 
and returning one stacked image (see issue #3)

Once this is done, then update average_trials to call this instead
"""
def stack_phosphenes(df):
    """
    Stacks drawings even when the number of phosphenese is inconsistent across drawings
    Only works for single electrode stimulation that generates 1 or 2 phosphene(s) in each drawing
    Input: A data frame that contains at least 5 columns: "subject" (string), "electrode" (string), "amplitude" (double), "frequency" (double), "image" (array)
        Having two additional columns "centroid1" (tuple) and "centroid2" (tuple) that describes the centroids of two phosphenes
        will shrink the running time significantly
    Output: A dataframe that contains 6 columns: "subject", "electrode", "amplitude", "frequency" "phosphene1_avg" (array) and "phosphene2_avg" (array)
    """

    df = df.reset_index(drop=True)

    # find the centroid of phosphene(s) in each drawing
    if ('centroid1' not in df.columns) or ('centroid2' not in df.columns):
        lst1 = []
        lst2 = []
        num_regions = []
        for i in range(len(df)):
            label_img = label(df['image'][i], connectivity = df['image'][i].ndim)
            props = regionprops(label_img)
            num_regions.append(len(props))
            lst1.append(props[0].centroid)
            if len(props) > 1:
                lst2.append(props[1].centroid)
            else:
                lst2.append('')

        df['centroid1'] = lst1
        df['centroid2'] = lst2
        df['num_regions'] = num_regions

    '''
    Group drawings by subject, amplitude, frequency, and electrode. Create two lists, the first list contains the first 
    stacked phosphene, and the second list contains the second stacked phosphene (if exists)
    Within each drawing:
    move the first phosphene to the center of the canvas
    move the second phosphene to the center of another canvas (if exists)
    Within each group: 
    if each drawing has 1 phosphene: stack them together, append the stacked image to list1, and append an empty array to list2;
    if each drawing has 2 phosphene: stack the first phosphene together and append the avergaed image to list1, stack the 
        second phosphene together and append the averaged image to list2;
    if some drawings have 1 phosphene and others have 2:
        1. average the centroids of all single-phosphene drawings in this group
        2. for each double-phosphene drawing, compare the centroids of two phosphenes to this averaged centroid
        3. for each double-phosphene drawing, if the first phosphene has a similar centroid to the averaged single-phosphene 
        centroid, stack the first phosphene to other single phosphene drawings. Then, stack all second phosphenes together with
        empty arrays (the number of empty arrays equals to the number of single-phosphene drawings)
        
        (For example, 1 double-phosphene drawing and 4 single-phosphene drawings in one group:
        get the centroid of this double-phosphene drawing (x1,y1) and (x2,y2)
        get the averaged centroid of these 4 single phosphenes (x_single_average,y_single_average)
        if (x1,y1) is closer to (x_single_average,y_single_average), stack the first phosphene to other single-phosphene drawings
        list1.append(array(single phosphene + single phosphene + single phosphene + single phosphene + 1st phosphene of the double-phosphene drawing))
        list2.append(array(empty array + empty array + empty array + empty array + 2rd phosphene of the double-phosphene drawing))
        )
    '''
    df_temp = df[['subject','amplitude','frequency','electrode','centroid1','centroid2']].reset_index(drop=True)
    x = []
    y = []
    for i in range(len(df_temp)):
        x.append(df_temp['centroid1'][i][0])
        y.append(df_temp['centroid1'][i][1])
    df_temp['x_avg'] = x
    df_temp['y_avg'] = y
    df_temp = (df_temp[df_temp['centroid2'] == '']).drop(columns = ['centroid2'])
    df1 = df_temp.groupby(['subject','amplitude','frequency','electrode']).mean()
    df = df.merge(df1, on=['subject','amplitude','frequency','electrode'])

    df['label'] = 1
    df['group'] = ''
    for i in range(len(df)):
        df['group'][i] = df['subject'][i] + '_' + df['electrode'][i] + '_' + str(df['amplitude'][i]) + '_' + str(df['frequency'][i])
        if df['centroid2'][i] != '':
            label1 = np.mean([df['centroid1'][i][0]-df['x_avg'][i], df['centroid1'][i][1]-df['y_avg'][i]])
            label2 = np.mean([df['centroid2'][i][0]-df['x_avg'][i], df['centroid2'][i][1]-df['y_avg'][i]])
            if abs(label1) < abs(label2):
                df['label'][i] = 1
            else:
                df['label'][i] = 2   
    df1 = df[['subject','amplitude','frequency','electrode','group']].drop_duplicates()

    empty_array = np.zeros((len(df['image'][0]),len(df['image'][0][0])))
    stacked_image = []
    for i in range(len(np.unique(df['group']))):
        img_list1 = []
        img_list2 = []
        sub = df[(df.group == np.unique(df['group'])[i])].reset_index(drop=True)
        for j in range(len(sub)):
            if sub['num_regions'][j] == 1:
                img_list1.append(center_image(label(sub['image'][j]) == 1))
                img_list2.append(empty_array)
            else:
                if sub['label'][j] == 1:
                    img_list1.append(center_image(label(sub['image'][j]) == 1))
                    img_list2.append(center_image(label(sub['image'][j]) == 2))
                else:
                    img_list1.append(center_image(label(sub['image'][j]) == 2))
                    img_list2.append(center_image(label(sub['image'][j]) == 1))
        stacked_image.append([np.mean(img_list1,axis=0),np.mean(img_list2,axis=0),np.unique(df['group'])[i]])
    stacked_image = pd.DataFrame(stacked_image,columns = ['phosphene1_avg','phosphene2_avg','group'])
    stacked_image = (stacked_image.merge(df1, on=['group'])).drop(columns = ['group'])
    return stacked_image

def find_matching_image(image, df):
    """
    Find the index of the matching image in the new dataframe
    Images don't always line up exactly, so it actually finds the
    image with the smallest number of pixel mismatches (usually < 5 for matches)
    """
    def key_img_match(k):
        new_img = df.loc[k, 'image']
        if image.shape != new_img.shape:
            return 999999999
        return np.sum(image != new_img)
    idx_matching = min(df.index, key = key_img_match)
    return idx_matching

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