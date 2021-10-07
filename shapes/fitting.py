import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from skimage.measure import label, regionprops
from skimage.transform import resize

from pulse2percept.models import BiphasicAxonMapModel, AxonMapModel
from pulse2percept.model_selection import ParticleSwarmOptimizer
from pulse2percept.implants import ArgusII
from pulse2percept.stimuli import BiphasicPulseTrain, Stimulus



class BiphasicAxonMapEstimator(BaseEstimator):
    """
    Estimates parameters for a BiphasicAxonMapModel
    
    Parameters:
    -------------
    implant : p2p.implant
        A patient specific implant. Will default to ArgusII with no rotation.
    relative_weight : float, optional
        Weight of size vs eccentricity for MSE loss function. Since size is much larger,
        (especially when squared) this value defaults to be very small (10e-6)
    """
    def __init__(self, verbose=True, implant=None, relative_weight=10e-6, **kwargs):
        self.verbose = verbose
        # Default to Argus II if no implant provided
        self.implant = implant
        if self.implant is None:
            self.implant = ArgusII()
        self.model = BiphasicAxonMapModel(xystep=0.5)
        self.model.build()
        self.relative_weight = relative_weight
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
        self.set_params(**kwargs)


    def get_params(self, deep=False):
        params = {
            attr : getattr(self.model, attr) for attr in \
                ['rho', 'axlambda', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9']
        }
        return params

    def get_props(self, drawing):
        """
        Returns the regionprops region with the largest area
        """
        props = regionprops(label(drawing))
        if len(props) == 0:
            return None
        return max(props, key=lambda x : x.area)

    def fit_size_model(self, df):
        """
        Estimates size_model parameters (rho scaling) of a BiphasicAxonMapModel
        based on drawings and amplitudes. 
        """
        amps = np.array(df['amp']).reshape(-1, 1)
        sizes = [self.get_props(image).area for image in df['image']]
        if len(np.unique(amps)) < 2:
            print("Warning: Not enough uniqiue amps to fit effects model, using defaults")
            return

        # fit linear regression
        lr = LinearRegression()
        lr.fit(amps, sizes)
        # rescale so that amp=1 is normalized
        s_norm = lr.predict(np.array(1).reshape(1, -1))[0]
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


    def fit(self, X, y=None, **fit_params):
        self.model.set_params(fit_params)
        # update model with params sent to us from PSO
        self.model.set_params(self.get_params())

        self.model.build()
        return self

    def predict(self, X):
        y_pred = []
        for row in X.itertuples():
            self.implant.stim = Stimulus({row.electrode1 : BiphasicPulseTrain(row.freq, row.amp1, row.pdur)})
            percept = self.model.predict_percept(self.implant)
            y_pred.append(percept)
        return pd.DataFrame(y_pred, index=X.index, columns=['predicted_image'])

    def score(self, X, y):
        y_pred = self.predict(X)

        for i in range(len(y_pred)):
            if y_pred[i].shape != y[i].shape():
                # Not the same image size, so phosphene size comparison doesnt mean anything
                y_pred[i] = resize(y_pred[i], y[i].shape)

        # can easily change this to be pixel MSE, MS-SSIM, etc
        # For now, weighted MSE between eccentricity and size
        pred_moments = np.array([[prop.area, prop.eccentricity] for prop in [self.get_props(p) for p in y_pred]])
        y_moments = np.array([[prop.area, prop.eccentricity] for prop in [self.get_props(p) for p in y]])

        score = np.mean(self.relative_weight * (pred_moments[:, 0] - y_moments[:, 0])**2 + (pred_moments[:, 1] - y_moments[:, 1])**2)

        if self.verbose:
            print('rho=%f, axlambda=%f, a5=%f, a6=%f, score=%f' % 
                                            (self.model.rho,
                                             self.model.axlambda,
                                             self.model.a5,
                                             self.model.a6,
                                             score))

        return score
        
        



    

