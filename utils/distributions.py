import numpy as np
from scipy.special import erf
from scipy.stats import rv_continuous


class FlippedGaussian(rv_continuous):
    def __init__(self, theta, *args, mu=0, sigma=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._theta = theta
        self._mu = mu
        self._sigma = sigma

    def _pdf(self, x, **kwargs) -> float:
        # NB: x should belong to [-1, 1]
        pi = np.pi
        sqrt = np.sqrt
        exp = np.exp
        c = sqrt(2*pi)
        mu = self._mu
        sigma = self._sigma
        theta = self._theta

        f0 = exp(-(mu**2)/(2*sigma**2))

        num = theta + (c / (sigma * 2 * pi)) * (f0 - exp(-(x - mu)**2/(2 * sigma**2)))
        den = (2*theta
               + erf(sqrt(2)*mu/(2*sigma) - sqrt(2)/(2*sigma))/2
               - erf(sqrt(2)*mu/(2*sigma) + sqrt(2)/(2*sigma))/2
               + sqrt(2)*exp(-mu**2/(2*sigma**2))/(sqrt(pi)*sigma))

        return num / den


class CustomDistribution(rv_continuous):
    def __init__(self, k, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if k < 0:
            raise ValueError(f"Parameter k must be positive, found {k}")

        if k > 4:
            raise ValueError(f"Parameter k must be less than 4, found {k}")

        self._k = k

    def _pdf(self, x, *args):
        a = self._k
        b = 1 - 3*a/4

        if -1 <= x <= -0.5:
            return -a*x + b
        elif 0.5 <= x <= 1:
            return a*x + b
        else:
            return 0

