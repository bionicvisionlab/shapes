import numpy as np
import pandas as pd
import math
import time
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from skimage.measure import label, regionprops
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler

from pulse2percept.models import BiphasicAxonMapModel, AxonMapModel
from pulse2percept.model_selection import ParticleSwarmOptimizer
from pulse2percept.implants import ArgusII
from pulse2percept.stimuli import BiphasicPulseTrain, Stimulus

import matplotlib.pyplot as plt


class BiphasicAxonMapEstimator(BaseEstimator):
    """
    Estimates parameters for a BiphasicAxonMapModel
    
    Parameters:
    -------------
    implant : p2p.implant, optional
        A patient specific implant. Will default to ArgusII with no rotation.
    model : p2p.models.BiphasicAxonMapModel, optional
        A patient specific model.
    mse_params: list of skimage regionprops params, optional
        The parameters to use to compute MSE. Each must be a valid parameter of skimage.measure.regionprops
        Defaults to ellipse major and minor axis.
        Note that due to scale differences between parameters, each will be transformed to have mean 0 and variance 1
    feature_importance : list of floats, optional
        Relative MSE weight for each loss param. 
        Note that this applies AFTER scaling mean to 0 and variance to 1 (if scale_features is true). 
        Thus, a value of 2 means the corresponding feature is twice as important as any feature with value 1
        Defaults to equal weighting
    scale_features : bool, optional
        Whether or not to scale features to have 0 mean and 1 variance. 
        Defaults to true
        Warning: If this is false, then feature_importances should be updated to reflect the difference in scale
        between different features
    """
    def __init__(self, verbose=True, mse_params=None, feature_importance=None, implant=None, model=None, resize=True, scale_features=True, **kwargs):
        self.verbose = verbose
        # Default to Argus II if no implant provided
        self.implant = implant
        if self.implant is None:
            self.implant = ArgusII()
        self.model = model
        if self.model is None:
            self.model = BiphasicAxonMapModel(xystep=0.5)
        self.model.build()
        self.mse_params = mse_params
        if self.mse_params is None:
            self.mse_params = ['major_axis_length', 'minor_axis_length']
        if feature_importance is not None:
            if len(feature_importance) != len(self.mse_params):
                raise ValueError("Feature_importance must be same length as mse_params: %d" % len(mse_params))
            self.feature_importance = feature_importance
        else:
            self.feature_importance = np.ones(len(self.mse_params))
        # Create all parameters here, but don't neccesarily need to use them all
        self.rho = self.model.rho
        self.axlambda = self.model.axlambda
        self.a0 = self.model.a0
        self.a1 = self.model.a1
        self.a2 = self.model.a2
        self.a3 = self.model.a3
        self.a4 = self.model.a4
        self.a5 = self.model.a5
        self.a6 = self.model.a6
        self.a7 = self.model.a7
        self.a8 = self.model.a8
        self.a9 = self.model.a9
        self.resize = resize
        self.scaler = None
        self.yshape = None
        self.scale_features = scale_features
        self.set_params(**kwargs)


    def get_params(self, deep=False):
        params = {
            attr : getattr(self, attr) for attr in \
                ['rho', 'axlambda', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9']
        }
        return params

    def get_props(self, drawing, threshold=1):
        """
        Returns the regionprops region with the largest area
        """
        if threshold is not None:
            props = regionprops(label(drawing > threshold))
        else:
            props = regionprops(label(drawing))
        if len(props) == 0:
            return None
        return max(props, key=lambda x : x.area)

    def fit_size_model(self, df):
        """
        Estimates size_model parameters (rho scaling) of a BiphasicAxonMapModel, A5 and A6,
        based on drawings and amplitudes. 
        """
        self.fit_size_model(df['amp1'], df['image'])
    
    def fit_size_model(self, x, y):
        amps = np.array(x).reshape(-1, 1)
        sizes = [self.get_props(image, threshold=None).area for image in y]
        if len(np.unique(amps)) < 2:
            print("Warning: Not enough uniqiue amps to fit effects model, using defaults")
            return

        # fit linear regression
        lr = LinearRegression()
        lr.fit(amps, sizes)
        # rescale so that amp=1.25 is normalized
        s_norm = lr.predict(np.array(1.25).reshape(1, -1))[0]
        sizes_scaled = sizes / s_norm
        lr = LinearRegression()
        lr.fit(amps, sizes_scaled)
        # get and set a5, a6
        self.a6 = lr.intercept_
        self.a5 = lr.coef_[0]
        # propogate to model and rebuild
        self.model.a5 = self.a5
        self.model.a6 = self.a6
        self.model.build()

        if self.verbose:
            print('a5=%f, a6=%f' % (self.model.a5, self.model.a6))


    def fit(self, X, y=None, **fit_params):
        self.model.set_params(fit_params)
        # update model with params sent to us from PSO
        self.model.set_params(self.get_params())
        # Make sure 1.25 * a5 + a6 = 1
        self.a6 = 1 - 1.25 * self.a5
        self.model.a6 = self.a6
        self.model.build()
        return self

    def predict(self, X):
        y_pred = []
        for row in X.itertuples():
            self.implant.stim = Stimulus({row.electrode1 : BiphasicPulseTrain(row.freq, row.amp1, row.pdur, stim_dur=math.ceil(3*row.pdur))})
            percept = self.model.predict_percept(self.implant)
            y_pred.append(percept.data[:, :, 0])
        return pd.Series(y_pred, index=X.index)

    def score(self, X, y):
        y_pred = self.predict(X)
        # If yshape has a value, then moments have been precomputed
        if self.yshape is not None:
            y_moments = y
        else:
            props = [self.get_props(p, threshold=None) for p in y]
            y_moments = np.array([[prop[param] for param in self.mse_params] if prop is not None else [0.0 for param in self.mse_params] for prop in props])
            self.yshape = y[0].shape
            if self.scale_features:
                self.scaler = StandardScaler(copy=False)
                y_moments = self.scaler.fit_transform(y_moments)

        if self.resize:
            for i in y_pred.index:
                y_pred[i] = resize(y_pred[i], self.yshape)

        # LOSS
        # weighted MSE
        pred_props = [self.get_props(p) for p in y_pred]
        pred_moments = np.array([[prop[param] for param in self.mse_params] if prop is not None else [0.0 for param in self.mse_params] for prop in pred_props])
        pred_moments = self.scaler.transform(pred_moments)
        
        score_contrib = np.mean((pred_moments - y_moments)**2 , axis=0) * self.feature_importance
        score = np.sum(score_contrib)

        if self.verbose:
            mses = [str(self.mse_params[i]) + ": " + str(round(score_contrib[i], 3)) for i in range(len(score_contrib))]
            print(('rho=%f, axlambda=%f, a5=%f, a6=%f, null_props=%.1f, score=%f, mses: ' + str(mses)) % 
                                            (self.model.rho,
                                             self.model.axlambda,
                                             self.model.a5,
                                             self.model.a6,
                                             np.sum(pred_moments==0) / len(score_contrib),
                                             score))

        return score

    def precompute_moments(self, y, shape=None):
        if shape is not None:
            y = [resize(img, shape, anti_aliasing=True) for img in y]
        props = [self.get_props(p, threshold=None) for p in y]
        moments = np.array([[prop[param] for param in self.mse_params] if prop is not None else [0.0 for param in self.mse_params] for prop in props])
        self.yshape = y[0].shape

        if self.scale_features: 
            # fit the scaler
            self.scaler = StandardScaler(copy=False)
            moments = self.scaler.fit_transform(moments)
            means = [str(self.mse_params[i]) + ": " + str(round(self.scaler.mean_[i], 2)) for i in range(len(self.scaler.mean_))]
            stds = [str(self.mse_params[i]) + ": " + str(round(self.scaler.scale_[i], 2)) for i in range(len(self.scaler.scale_))]
            if self.verbose:
                print("Removing means ({}) and scaling standard deviations ({}) to be 1".format(means, stds))
        return moments
        
        
class AxonMapEstimator(BaseEstimator):
    """
    Estimates parameters for a AxonMapModel
    
    Parameters:
    -------------
    implant : p2p.implant, optional
        A patient specific implant. Will default to ArgusII with no rotation.
    model : p2p.models.AxonMapModel, optional
        A patient specific model.
    mse_params: list of skimage regionprops params, optional
        The parameters to use to compute MSE. Each must be a valid parameter of skimage.measure.regionprops
        Defaults to ellipse major and minor axis.
        Note that due to scale differences between parameters, each will be transformed to have mean 0 and variance 1
    feature_importance : list of floats, optional
        Relative MSE weight for each loss param. 
        Note that this applies AFTER scaling mean to 0 and variance to 1 (if scale_features is true). 
        Thus, a value of 2 means the corresponding feature is twice as important as any feature with value 1
        Defaults to equal weighting
    scale_features : bool, optional
        Whether or not to scale features to have 0 mean and 1 variance. 
        Defaults to true
        Warning: If this is false, then feature_importances should be updated to reflect the difference in scale
        between different features
    """
    def __init__(self, verbose=True, feature_importance=None, implant=None, model=None, resize=True, mse_params=None, scale_features=True, **kwargs):
        self.verbose = verbose
        # Default to Argus II if no implant provided
        self.implant = implant
        if self.implant is None:
            self.implant = ArgusII()
        self.model = model
        if self.model is None:
            self.model = AxonMapModel(xystep=0.5)
        self.model.build()
        self.mse_params = mse_params
        if self.mse_params is None:
            self.mse_params = ['major_axis_length', 'minor_axis_length']
        if feature_importance is not None:
            if len(feature_importance) != len(self.mse_params):
                raise ValueError("Feature_importance must be same length as mse_params: %d" % len(mse_params))
            self.feature_importance = feature_importance
        else:
            self.feature_importance = np.ones(len(self.mse_params))
        # Create all parameters here, but don't neccesarily need to use them all
        self.rho = self.model.rho
        self.axlambda = self.model.axlambda
        self.resize = resize
        self.scaler = None
        self.yshape = None
        self.scale_features = scale_features
        self.set_params(**kwargs)


    def get_params(self, deep=False):
        params = {
            attr : getattr(self, attr) for attr in \
                ['rho', 'axlambda']
        }
        return params

    def get_props(self, drawing, threshold=1):
        """
        Returns the regionprops region with the largest area
        """
        if threshold is not None:
            props = regionprops(label(drawing > threshold))
        else:
            props = regionprops(label(drawing))
        if len(props) == 0:
            return None
        return max(props, key=lambda x : x.area)

    def fit(self, X, y=None, **fit_params):
        self.model.set_params(fit_params)
        # update model with params sent to us from PSO
        self.model.set_params(self.get_params())
        self.model.build()
        return self

    def predict(self, X):
        y_pred = []
        for row in X.itertuples():
            self.implant.stim = Stimulus({row.electrode1 : BiphasicPulseTrain(row.freq, row.amp1, row.pdur, stim_dur=math.ceil(3*row.pdur))})
            percept = self.model.predict_percept(self.implant)
            y_pred.append(percept.get_brightest_frame())
        return pd.Series(y_pred, index=X.index)

    def score(self, X, y):
        y_pred = self.predict(X)
        # If yshape has a value, then moments have been precomputed
        if self.yshape is not None:
            y_moments = y
        else:
            props = [self.get_props(p, threshold=None) for p in y]
            y_moments = np.array([[prop[param] for param in self.mse_params] if prop is not None else [0.0 for param in self.mse_params] for prop in props])
            self.yshape = y[0].shape
            if self.scale_features:
                self.scaler = StandardScaler(copy=False)
                y_moments = self.scaler.fit_transform(y_moments)

        if self.resize:
            for i in y_pred.index:
                y_pred[i] = resize(y_pred[i], self.yshape)

        # LOSS
        # weighted MSE
        pred_props = [self.get_props(p) for p in y_pred]
        pred_moments = np.array([[prop[param] for param in self.mse_params] if prop is not None else [0.0 for param in self.mse_params] for prop in pred_props])
        pred_moments = self.scaler.transform(pred_moments)
        
        score_contrib = np.mean((pred_moments - y_moments)**2 , axis=0) * self.feature_importance
        score = np.sum(score_contrib)

        if self.verbose:
            mses = [str(self.mse_params[i]) + ": " + str(round(score_contrib[i], 3)) for i in range(len(score_contrib))]
            print(('rho=%f, axlambda=%f, null_props=%.1f, score=%f, mses: ' + str(mses)) % 
                                            (self.model.rho,
                                             self.model.axlambda,
                                             np.sum(pred_moments==0) / len(score_contrib),
                                             score))

        return score

    def precompute_moments(self, y, shape=None):
        if shape is not None:
            y = [resize(img, shape) for img in y]
        return np.array([[prop.area, prop.eccentricity] if prop is not None else [0.0, 0.0] for prop in [self.get_props(p, threshold=None) for p in y]])


    

