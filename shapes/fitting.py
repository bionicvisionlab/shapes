import numpy as np
import pandas as pd
import math
import time
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from skimage.measure import label, regionprops, moments_central
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
            self.mse_params = ["central_moments"]

        num_feats = len(self.mse_params)
        if "moments_central" in self.mse_params:
            num_feats += 6
        if feature_importance is not None:
            if len(feature_importance) != num_feats:
                raise ValueError("Feature_importance must be same length as mse_params: %d" % num_feats)
            self.feature_importance = feature_importance
        else:
            self.feature_importance = np.ones(num_feats)

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

    def get_props(self, drawing, threshold="compute"):
        """
        Returns the regionprops region with the largest area
        """
        if threshold == 'compute':
            threshold = (drawing.max() - drawing.min()) * 0.1 + drawing.min()
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

    def score(self, X, y, return_mses=False):
        y_pred = self.predict(X)
        # If yshape has a value, then moments have been precomputed
        if self.yshape is not None:
            y_moments = y
        else:
            raise ValueError("Please precompute the y features using estimator.compute_moments")
        pred_moments = self.compute_moments(y_pred, fit_scaler=False, shape=self.yshape, threshold="compute")

        if len(self.feature_importance) != len(self._mse_params):
            print("Warning, got different length feature_importances and mse_params. Did you set one manually?\n"
                "Defaulting to equal weighting")
            self.feature_importance = np.ones(len(self._mse_params))
        
        # LOSS
        # weighted MSE
        score_contrib = np.mean((pred_moments - y_moments)**2 , axis=0) * self.feature_importance
        score = np.sum(score_contrib)

        if self.verbose:
            mses = [str(self._mse_params[i]) + ": " + str(round(score_contrib[i], 1)) for i in range(len(score_contrib))]
            print(('score=%.3f, rho=%.1f, lambda=%.1f, a5=%.3f, a6=%.3f, mses: ' + str(mses)) % 
                                            (score,
                                             self.model.rho,
                                             self.model.axlambda,
                                             self.model.a5,
                                             self.model.a6))
        if not return_mses: 
            return score
        else:
            return score, score_contrib

    def compute_moments(self, y, fit_scaler=True, shape=None, threshold=None):
        if shape is not None:
            y = [resize(img, shape, anti_aliasing=True) for img in y]

        if self.mse_params != ['moments_central']:
            # Only need to compute props if we have more than moments_central
            props = [self.get_props(p, threshold=threshold) for p in y]
        else: 
            # Anything but None
            props = [0.0 for i in y]
        moments = [] # y feats
        new_mse_params = [] # new parameters after expanding moments
        for idx_prop, (prop, y_img) in enumerate(zip(props, y)):
            prop_moments = []
            if prop is not None:
                for idx_param, param in enumerate(self.mse_params):
                    if param != 'moments_central':
                        prop_moments.append(prop[param])
                        if idx_prop == 0:
                            new_mse_params.append(param)
                    else:
                        # Compute moments on whole image not prop
                        if threshold == "compute":
                            threshold = (y_img.max() - y_img.min()) * 0.1 + y_img.min()
                            central_moments = moments_central(y_img > threshold)
                        elif threshold is not None:
                            central_moments = moments_central(y_img > threshold)
                        else:
                            central_moments = moments_central(y_img)
                        for r in range(3):
                                for c in range(3):
                                    if r + c != 1:
                                        prop_moments.append(central_moments[r,c])
                                        if idx_prop == 0:
                                            new_mse_params.append("M" + str(r) + str(c))
                prop_moments = np.array(prop_moments)
            else:
                # all 0's, but how many depends on if central_moments is a param
                if "moments_central" not in self.mse_params:
                    prop_moments = np.zeros(len(self.mse_params))
                else:
                    prop_moments = np.zeros((len(self.mse_params) + 6))
            moments.append(prop_moments)

        moments = np.array(moments)
        self._mse_params = new_mse_params
        if shape is None:
            self.yshape = y[0].shape

        if self.scale_features and fit_scaler: 
            # fit the scaler
            self.scaler = StandardScaler(copy=False)
            moments = self.scaler.fit_transform(moments)
            means = [str(self._mse_params[i]) + ": " + str(round(self.scaler.mean_[i], 2)) for i in range(len(self.scaler.mean_))]
            stds = [str(self._mse_params[i]) + ": " + str(round(self.scaler.scale_[i], 2)) for i in range(len(self.scaler.scale_))]
            if self.verbose:
                print("Removing means ({}) \nScaling standard deviations ({}) to be 1".format(means, stds))
        elif self.scale_features:
            # just transform, dont fit
            moments = self.scaler.transform(moments)
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

    def get_props(self, drawing, threshold="compute"):
        """
        Returns the regionprops region with the largest area
        """
        if threshold == 'compute':
            threshold = (drawing.max() - drawing.min()) * 0.1 + drawing.min()
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

    def score(self, X, y, return_mses=False):
        y_pred = self.predict(X)
        # If yshape has a value, then moments have been precomputed
        if self.yshape is not None:
            y_moments = y
        else:
            raise ValueError("Please precompute the y features using estimator.compute_moments")
        pred_moments = self.compute_moments(y_pred, fit_scaler=False, shape=self.yshape, threshold="compute")

        if len(self.feature_importance) != len(self._mse_params):
            print("Warning, got different length feature_importances and mse_params. Did you set one manually?\n"
                "Defaulting to equal weighting")
            self.feature_importance = np.ones(len(self._mse_params))
        
        # LOSS
        # weighted MSE
        score_contrib = np.mean((pred_moments - y_moments)**2 , axis=0) * self.feature_importance
        score = np.sum(score_contrib)

        if self.verbose:
            mses = [str(self._mse_params[i]) + ": " + str(round(score_contrib[i], 1)) for i in range(len(score_contrib))]
            print(('score=%.3f, rho=%.1f, lambda=%.1f, mses: ' + str(mses)) % 
                                            (score,
                                             self.model.rho,
                                             self.model.axlambda))
        if not return_mses: 
            return score
        else:
            return score, score_contrib

    def compute_moments(self, y, fit_scaler=True, shape=None, threshold=None):
        if shape is not None:
            y = [resize(img, shape, anti_aliasing=True) for img in y]

        if self.mse_params != ['moments_central']:
            # Only need to compute props if we have more than moments_central
            props = [self.get_props(p, threshold=threshold) for p in y]
        else: 
            # Anything but None
            props = [0.0 for i in y]
        moments = [] # y feats
        new_mse_params = [] # new parameters after expanding moments
        for idx_prop, (prop, y_img) in enumerate(zip(props, y)):
            prop_moments = []
            if prop is not None:
                for idx_param, param in enumerate(self.mse_params):
                    if param != 'moments_central':
                        prop_moments.append(prop[param])
                        if idx_prop == 0:
                            new_mse_params.append(param)
                    else:
                        # Compute moments on whole image not prop
                        if threshold == "compute":
                            threshold = (y_img.max() - y_img.min()) * 0.1 + y_img.min()
                            central_moments = moments_central(y_img > threshold)
                        elif threshold is not None:
                            central_moments = moments_central(y_img > threshold)
                        else:
                            central_moments = moments_central(y_img)
                        for r in range(3):
                                for c in range(3):
                                    if r + c != 1:
                                        prop_moments.append(central_moments[r,c])
                                        if idx_prop == 0:
                                            new_mse_params.append("M" + str(r) + str(c))
                prop_moments = np.array(prop_moments)
            else:
                # all 0's, but how many depends on if central_moments is a param
                if "moments_central" not in self.mse_params:
                    prop_moments = np.zeros(len(self.mse_params))
                else:
                    prop_moments = np.zeros((len(self.mse_params) + 6))
            moments.append(prop_moments)

        moments = np.array(moments)
        self._mse_params = new_mse_params
        if shape is None:
            self.yshape = y[0].shape

        if self.scale_features and fit_scaler: 
            # fit the scaler
            self.scaler = StandardScaler(copy=False)
            moments = self.scaler.fit_transform(moments)
            means = [str(self._mse_params[i]) + ": " + str(round(self.scaler.mean_[i], 2)) for i in range(len(self.scaler.mean_))]
            stds = [str(self._mse_params[i]) + ": " + str(round(self.scaler.scale_[i], 2)) for i in range(len(self.scaler.scale_))]
            if self.verbose:
                print("Removing means ({}) \nScaling standard deviations ({}) to be 1".format(means, stds))
        elif self.scale_features:
            # just transform, dont fit
            moments = self.scaler.transform(moments)
        return moments


    

