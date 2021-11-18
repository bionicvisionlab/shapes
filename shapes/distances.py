"""
Code for measuring retinal and axonal distances
"""

import numpy as np
import pandas as pd

def get_closest_bundles(e1_x, e1_y, e2_x, e2_y, model):
    bundles = model.grow_axon_bundles()

    bundles1 = model.find_closest_axon(bundles, xret=e1_x, yret=e1_y)
    bundles2 = model.find_closest_axon(bundles, xret=e2_x, yret=e2_y)
    if len(e1_x) == 1:
        bundles1 = [bundles1]
        bundles2 = [bundles2]

    return bundles1, bundles2

def calc_bundle_tangent(bundles, x, y):
    tangents = []
    for bundle, xc, yc in zip(bundles, x, y):
        idx = np.argmin((bundle[:, 0] - xc) ** 2 + (bundle[:, 1] - yc) ** 2)
        # Calculate orientation from atan2(dy, dx):
        if idx == 0:
            # Bundle index 0: there's no index -1
            dx = bundle[1, :] - bundle[0, :]
        elif idx == bundle.shape[0] - 1:
            # Bundle index -1: there's no index len(bundle)
            dx = bundle[-1, :] - bundle[-2, :]
        else:
            # Else: Look at previous and subsequent segments:
            dx = (bundle[idx + 1, :] - bundle[idx - 1, :]) / 2
        dx[1] *= -1
        tangent = np.arctan2(*dx[::-1])
        # # Confine to (-pi/2, pi/2):
        if tangent < np.deg2rad(-90):
            tangent += np.deg2rad(180)
        if tangent > np.deg2rad(90):
            tangent -= np.deg2rad(180)
        tangents.append(tangent)
    return np.array(tangents)

def dist_perpendicular_tangential(electrode1, electrode2, implant, model, strategy="average"):
    """ Calculates the retinal distance, and decomposes retinal distance into 
    components perpendicular and tangential to the axons at each electrode
    Parameters:
    -----------
    electrode1, electrode2 : list of str or int
        Electrode name in implant
    model : p2p.models.AxonMapModel
        Model used to obtain axon bundles
    implant : p2p.implants.*
        Implant to use
    
    Returns:
    ---------
    Array with shape (N, 3), where the columns are retinal distance, 
    perpendicular distance, and tangential distance
    """
    if not isinstance(electrode1, (list, np.ndarray, pd.Series)):
        electrode1 = [electrode1]
    if not isinstance(electrode2, (list, np.ndarray, pd.Series)):
        electrode2 = [electrode2]

    e1_x = np.array([implant.electrodes[e].x for e in electrode1])
    e1_y = np.array([implant.electrodes[e].y for e in electrode1])
    e2_x = np.array([implant.electrodes[e].x for e in electrode2])
    e2_y = np.array([implant.electrodes[e].y for e in electrode2])

    bundles1, bundles2 = get_closest_bundles(e1_x, e1_y, e2_x, e2_y, model)
    tangents1 = calc_bundle_tangent(bundles1, e1_x, e1_y)
    tangents2 = calc_bundle_tangent(bundles2, e2_x, e2_y)

    ret_distance = np.stack([e1_x - e2_x, e1_y - e2_y], axis=1)
    vec1 = np.stack([-np.cos(tangents1), np.sin(tangents1)], axis=1)
    vec2 = np.stack([-np.cos(tangents2), np.sin(tangents2)], axis=1)
    dtan1 = np.abs(np.sum(ret_distance * vec1, axis=1))
    dtan2 = np.abs(np.sum(ret_distance * vec2, axis=1))

    dret = np.linalg.norm(ret_distance, axis=1)

    dperp1 = np.sqrt(dret**2 - dtan1**2)
    dperp2 = np.sqrt(dret**2 - dtan2**2)

    dtan = (dtan1 + dtan2) / 2
    dperp = (dperp1 + dperp2) / 2
    
    if strategy == "average":
        return np.stack([dret, dperp, dtan], axis=1)
    elif strategy == "upstream":
        use_e1 = e1_x >= e1_y
        dists = np.zeros((len(e1_x), 3))
        dists[use_e1] = np.stack([dret[use_e1], dperp1[use_e1], dtan1[use_e1]], axis=1)
        dists[~use_e1] = np.stack([dret[~use_e1], dperp2[~use_e1], dtan2[~use_e1]], axis=1)
        return dists
    else:
        raise NotImplementedError()


def dist_across_along_same(e1x, e1y, e2x, e2y, bundle1, bundle2):
    # dist across and along when the electrodes are on the same hemisphere
    # assume that electrode1 is on the right. if its not, then switch and call again
    if e1x < e2x :
        return dist_across_along_same(e2x, e2y, e1x, e1y, bundle2, bundle1)
    
    # find the closest point on bundle 2 to e1. This should always be closer than e2
    seg_dists = np.sqrt((bundle2[:, 0] - e1x)**2 + (bundle2[:, 1] - e1y)**2)
    idx_closest1 = np.argmin(seg_dists)
    d_across = seg_dists[idx_closest1]
    # find closest one to e2
    idx_closest2 = np.argmin((bundle2[:, 0] - e2x) ** 2 +
                            (bundle2[:, 1] - e2y) ** 2)
    # if this fails then the assumption that e1 will always be closer to the axon than to the electrode is false
    assert idx_closest1 <= idx_closest2

    # now find the distance along the axon from that point to e2
    d_along = np.sum(np.sqrt(np.diff(bundle2[idx_closest1:idx_closest2, 0], axis=0) ** 2 + \
                        np.diff(bundle2[idx_closest1:idx_closest2, 1], axis=0) ** 2)) 
    d_ret = np.sqrt((e1x - e2x)**2 + (e1y-e2y)**2)

    return [d_ret, d_across, d_along]


def dist_across_along_axonal(e1x, e1y, e2x, e2y, bundle1, bundle2):    
    # extend streaks all the way to raphe and find the distance in between 
    # find closest point on b1 to e1 and b2 to e2
    idx_closest1 = np.argmin((bundle1[:, 0] - e1x) ** 2 +
                            (bundle1[:, 1] - e1y) ** 2)
    idx_closest2 = np.argmin((bundle2[:, 0] - e2x) ** 2 +
                            (bundle2[:, 1] - e2y) ** 2)
    d1 = np.sum(np.sqrt(np.diff(bundle1[idx_closest1:, 0], axis=0) ** 2 + \
                        np.diff(bundle1[idx_closest1:, 1], axis=0) ** 2)) 
    d2 = np.sum(np.sqrt(np.diff(bundle2[idx_closest2:, 0], axis=0) ** 2 + \
                        np.diff(bundle2[idx_closest2:, 1], axis=0) ** 2)) 
    d_along = d1 + d2
    d_across = np.sqrt((bundle1[-1, 0] - bundle2[-1, 0])**2 + (bundle1[-1, 1] - bundle2[-1, 1])**2)
    d_ret = np.sqrt((e1x - e2x)**2 + (e1y-e2y)**2)
    return [d_ret, d_across, d_along]

def dist_across_along_radial(e1x, e1y, e2x, e2y, bundle1, bundle2):
    # extend radial current spread until you hit the other axon
    # assume that electrode1 is on the right. if its not, then switch and call again
    if e1x < e2x :
        return dist_across_along_radial(e2x, e2y, e1x, e1y, bundle2, bundle1)

    # if we chop off e2 past the electrode, this is exactly the same as dist_across_along_same
    idx_closest2 = np.argmin((bundle2[:, 0] - e2x) ** 2 +
                            (bundle2[:, 1] - e2y) ** 2)
    bundle2 = bundle2[:idx_closest2]
    return dist_across_along_same(e1x, e1y, e2x, e2y, bundle1, bundle2)

def dist_across_along(electrode1, electrode2, implant, model, strategy="axonal"):
    """ Calculates the retinal distance, and decomposes retinal distance into 
    components perpendicular and tangential to the axons at each electrode
    Parameters:
    -----------
    electrode1, electrode2 : list of str or int
        Electrode name in implant
    model : p2p.models.AxonMapModel
        Model used to obtain axon bundles
    implant : p2p.implants.*
        Implant to use
    
    Returns:
    ---------
    Array with shape (N, 3), where the columns are retinal distance, 
    distance across axons, and distance between axons
    """
    if not isinstance(electrode1, (list, np.ndarray, pd.Series)):
        electrode1 = [electrode1]
    if not isinstance(electrode2, (list, np.ndarray, pd.Series)):
        electrode2 = [electrode2]

    e1_x = np.array([implant.electrodes[e].x for e in electrode1])
    e1_y = np.array([implant.electrodes[e].y for e in electrode1])
    e2_x = np.array([implant.electrodes[e].x for e in electrode2])
    e2_y = np.array([implant.electrodes[e].y for e in electrode2])

    bundles1, bundles2 = get_closest_bundles(e1_x, e1_y, e2_x, e2_y, model)

    distances = []
    for idx, (e1x, e1y, e2x, e2y, bundle1, bundle2) in enumerate(zip(e1_x, e1_y, e2_x, e2_y, bundles1, bundles2)):
        if np.sign(e1y) == np.sign(e2y):
            # same hemisphere
            distances.append(dist_across_along_same(e1x, e1y, e2x, e2y, bundle1, bundle2))
        else:
            if strategy == "axonal":
                distances.append(dist_across_along_axonal(e1x, e1y, e2x, e2y, bundle1, bundle2))
            elif strategy == 'radial':
                distances.append(dist_across_along_radial(e1x, e1y, e2x, e2y, bundle1, bundle2))
            elif strategy == 'mixed':
                if np.sign(e1x) == 1 and np.sign(e2x) == 1:
                    # both electrodes in quadrant II/IV
                    distances.append(dist_across_along_radial(e1x, e1y, e2x, e2y, bundle1, bundle2))
                else:
                    # one of electrodes is in quadrants II/III
                    distances.append(dist_across_along_axonal(e1x, e1y, e2x, e2y, bundle1, bundle2))
            else:
                raise NotImplementedError()
    return np.array(distances)