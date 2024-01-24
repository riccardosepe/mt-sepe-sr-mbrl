import sys
from typing import Tuple

import torch


# TODO
def sign(a: float, b: float):
    return (a if a >= 0 else -a) if b >= 0 else (-a if a >= 0 else a)


class StepperBase:
    x: float
    xold: float
    y: torch.Tensor
    dydx: torch.Tensor
    atol: float
    rtol: float
    dense: bool
    hdid: float
    hnext: float
    EPS: float
    n: int
    neqn: int
    yout: torch.Tensor
    yerr: torch.Tensor

    def __init__(self, yy, dydxx, xx, atoll, rtoll, dens):
        self.x = xx
        self.y = yy
        self.dydx = dydxx
        self.atol = atoll
        self.rtol = rtoll
        self.dense = dens
        self.n = len(self.y)
        self.neqn = self.n
        self.yout = torch.empty((self.n,))
        self.yerr = torch.empty((self.n,))


class Output:
    kmax: int
    nvar: int
    nsave: int
    dense: bool
    count: int
    x1: float
    x2: float
    xout: float
    dxout: float
    xsave: torch.Tensor
    ysave: torch.Tensor

    def __init__(self, nsavee: int = None):
        if nsavee is None:
            self.kmax = -1
            self.dense = False
            self.count = 0
        else:
            self.kmax = 500
            self.nsave = nsavee
            self.count = 0
            self.xsave = torch.empty((self.kmax,))
            self.dense = bool(nsavee > 0)

    def init(self, neqn: int, xlo: float, xhi: float):
        self.nvar = neqn
        if self.kmax == -1:
            return
        self.ysave = self.ysave.resize(self.nvar, self.kmax)
        if self.dense:
            self.x1 = xlo
            self.x2 = xhi
            self.xout = self.x1
            self.dxout = (self.x2 - self.x1) / self.nsave

    def resize(self):
        kold = self.kmax
        self.kmax *= 2
        tempvec = self.xsave.clone()
        # double xsave length
        self.xsave.resize(self.kmax)
        for k in range(kold):
            self.xsave[k] = tempvec[k]
        tempmat = self.ysave.clone()
        self.ysave.resize(self.nvar, self.kmax)
        for i in range(self.nvar):
            for k in range(kold):
                self.ysave[i][k] = tempmat[i][k]

    # I think the following three functions are only useful when dense=True
    def save_dense(self, s: StepperBase, xout: float, h: float):
        pass

    def save(self, x: float, y: torch.Tensor):
        if self.kmax <= 0:
            return
        if self.count == self.kmax:
            self.resize()
        for i in range(self.nvar):
            self.ysave[i][self.count] = y[i]
        self.count += 1
        self.xsave[self.count] = x

    def out(self, nstp: int, x: float, y: torch.Tensor, stepper, h: float):
        raise NotImplementedError("Dense should never be used")


class OdeInt:
    MAXSTP = 50000
    EPS: float
    nok: int
    nbad: int
    nvar: int
    x1: float
    x2: float
    hmin: float
    dense: bool
    y: torch.Tensor
    dydx: torch.Tensor
    ystart: torch.Tensor
    out: Output
    derivs: callable
    s: StepperBase
    nstp: int
    x: float
    h: float

    def __init__(self,
                 ystartt: torch.Tensor,
                 xx1: float,
                 xx2: float,
                 atol: float,
                 rtol: float,
                 h1: float,
                 hminn: float,
                 outt: Output,
                 derivss: callable):
        self.nvar = len(ystartt)
        self.y = torch.empty((self.nvar, ))
        self.dydx = torch.empty((self.nvar, ))
        self.ystart = ystartt
        self.x = xx1
        self.nok = 0
        self.nbad = 0
        self.x1 = xx1
        self.x2 = xx2
        self.hmin = hminn
        self.dense = outt.dense
        self.out = outt
        self.derivs = derivss
        self.s = StepperDopr5(self.y, self.dydx, self.x,
                              atol, rtol, self.dense)
        self.EPS = sys.float_info.min
        self.h = sign(h1, self.x2 - self.x1)
        for i in range(self.nvar):
            self.y[i] = self.ystart[i]
        self.out.init(self.s.neqn, self.x1, self.x2)

    def integrate(self):
        self.s: StepperDopr5
        self.dydx = self.derivs(self.x, self.y).clone()
        if self.dense:
            self.out.out(-1, self.x, self.y, self.s, self.h)
        else:
            self.out.save(self.x, self.y)
        for self.nstp in range(self.MAXSTP):
            if (self.x + self.h*1.001-self.x2) * (self.x2 - self.x1) > 0.:
                self.h = self.x2 - self.x
            self.s.step(self.h, self.derivs)
            if self.s.hdid == self.h:
                self.nok += 1
            else:
                self.nbad += 1

            if self.dense:
                raise NotImplementedError
            else:
                self.out.save(self.x, self.y)

            if (self.x - self.x2) * (self.x2 - self.x1) >= 0.:
                self.ystart = self.y.clone()
                if self.out.kmax > 0 and abs(self.out.xsave[self.out.count-1]-self.x2) > 100. * abs(self.x2)*self.EPS:
                    self.out.save(self.x, self.y)
                return
            if abs(self.s.hnext) <= self.hmin:
                raise RuntimeError("Step size too small in OdeInt")
            self.h = self.s.hnext
        raise RuntimeError("Too many steps in OdeInt")


class Controller:
    hnext: float
    errold: float
    reject: bool

    def __init__(self, ):
        self.reject = False
        self.errold = 1e-4

    def success(self, err: float, h: float) -> Tuple[float, bool]:
        beta = 0.
        alpha = 0.2 - beta * 0.75
        safe = 0.9
        minscale = 0.2
        maxscale = 10.0

        scale: float
        if err <= 1.:
            if err == 0.0:
                scale = maxscale
            else:
                scale = safe * err ** (-alpha) * self.errold ** beta
                if scale < minscale:
                    scale = minscale
                if scale > maxscale:
                    scale = maxscale

            if self.reject:
                self.hnext = h * min(scale, 1.0)
            else:
                self.hnext = h * scale

            self.errold = max(err, 1e-4)
            self.reject = False
            return h, True

        else:
            scale = max(safe * err ** (-alpha), minscale)
            h *= scale
            self.reject = True
            return h, False


class StepperDopr5(StepperBase):
    k2: torch.Tensor
    k3: torch.Tensor
    k4: torch.Tensor
    k5: torch.Tensor
    k6: torch.Tensor
    rcont1: torch.Tensor
    rcont2: torch.Tensor
    rcont3: torch.Tensor
    rcont4: torch.Tensor
    rcont5: torch.Tensor
    dydxnew: torch.Tensor

    def __init__(self, yy, dydxx, xx, atoll, rtoll, dens):
        super().__init__(yy, dydxx, xx, atoll, rtoll, dens)
        self.k2 = torch.empty((self.n,))
        self.k3 = torch.empty((self.n,))
        self.k4 = torch.empty((self.n,))
        self.k5 = torch.empty((self.n,))
        self.k6 = torch.empty((self.n,))
        self.rcont1 = torch.empty((self.n,))
        self.rcont2 = torch.empty((self.n,))
        self.rcont3 = torch.empty((self.n,))
        self.rcont4 = torch.empty((self.n,))
        self.rcont5 = torch.empty((self.n,))
        self.dydxnew = torch.empty((self.n,))

        self.EPS = sys.float_info.min  # smallest possible double precision floating point

        self.con = Controller()

    def step(self, htry: float, derivs):
        h = htry
        while True:
            self.dy(h, derivs)  # TODO
            err = self.error()
            print(err)
            h, success = self.con.success(err, h)
            if success:
                break
            if abs(h) <= abs(self.x) * self.EPS:
                raise RuntimeError("Stepsize underflow in StepperDopr5")
        # Following part should be useless
        if self.dense:
            self.prepare_dense(h, derivs)
        self.dydx = self.dydxnew
        self.y = self.yout
        self.xold = self.x
        self.hdid = h
        self.x += self.hdid
        self.hnext = self.con.hnext

    def dy(self, h: float, derivs):
        c2 = 0.2
        c3 = 0.3
        c4 = 0.8
        c5 = 8.0 / 9.0
        a21 = 0.2
        a31 = 3.0 / 40.0
        a32 = 9.0 / 40.0
        a41 = 44.0 / 45.0
        a42 = -56.0 / 15.0
        a43 = 32.0 / 9.0
        a51 = 19372.0 / 6561.0
        a52 = -25360.0 / 2187.0
        a53 = 64448.0 / 6561.0
        a54 = -212.0 / 729.0
        a61 = 9017.0 / 3168.0
        a62 = -355.0 / 33.0
        a63 = 46732.0 / 5247.0
        a64 = 49.0 / 176.0
        a65 = -5103.0 / 18656.0
        a71 = 35.0 / 384.0
        a73 = 500.0 / 1113.0
        a74 = 125.0 / 192.0
        a75 = -2187.0 / 6784.0
        a76 = 11.0 / 84.
        e1 = 71.0 / 57600.0
        e3 = -71.0 / 16695.0
        e4 = 71.0 / 1920.0
        e5 = -17253.0 / 339200.0
        e6 = 22.0 / 525.0
        e7 = -1.0 / 40.0

        ytemp = self.y + h*a21*self.dydx

        self.k2 = derivs(self.x + c2 * h, ytemp).clone()  # second step

        ytemp = self.y + h * (a31 * self.dydx + a32 * self.k2)

        self.k3 = derivs(self.x + c3 * h, ytemp).clone()

        ytemp = self.y + h * (a41 * self.dydx + a42 * self.k2 + a43 * self.k3)

        self.k4 = derivs(self.x + c4 * h, ytemp).clone()

        ytemp = self.y + h * (a51 * self.dydx + a52 *
                              self.k2 + a53 * self.k3 + a54 * self.k4)

        self.k5 = derivs(self.x + c5 * h, ytemp).clone()

        ytemp = self.y + h * (a61 * self.dydx + a62 * self.k2 +
                              a63 * self.k3 + a64 * self.k4 + a65 * self.k5)

        xph = self.x + h
        self.k6 = derivs(xph, ytemp).clone()

        self.yout = self.y + h * \
            (a71 * self.dydx + a73 * self.k3 + a74 *
             self.k4 + a75 * self.k5 + a76 * self.k6)

        self.dydxnew = derivs(xph, self.yout).clone()
        self.yerr = h * (e1 * self.dydx + e3 * self.k3 + e4 *
                         self.k4 + e5 * self.k5 + e6 * self.k6 + e7 * self.dydxnew)

    def prepare_dense(self, h: float, derivs):
        pass

    def dense_out(self, i: int, x: float, h: float) -> float:
        pass

    def error(self):
        err = 0.
        for i in range(self.n):
            sk = self.atol + self.rtol * \
                max(torch.abs(self.y[i]), torch.abs(self.yout[i]))
            err += (self.yerr[i] / sk) ** 2
        return (err / self.n) ** 0.5


def f(t, y):
    return y


if __name__ == '__main__':
    y0 = torch.tensor([1.])
    t0 = 0.
    t1 = 1.
    out = Output()
    odeint = OdeInt(y0, t0, t1, 1e-3, 1e-6, h1=1e-1,
                    hminn=1e-2, outt=out, derivss=f)
    odeint.integrate()
    print(odeint.out)
