ύ      }(H
dill._dill_create_function(h_create_code(KK K KKKCB   | \}}}}}}}}}	}
}}}|d }t |
}|| }|| }|
| }|t | }t|
}|d | }d| | d| |d   |	| | |  |	| ||   d| |d | |d |    d| || ||  d | | | || t|  d    S NKG?ΰ      cossin(	_Dummy_34m1l1r1I1m2l2r2I2gq1q2q1dotq2dotx0x1x2x3x4x5x6x7t<lambdifygenerated-1>_lambdifygeneratedKC ))tR}__name__Nsh$NNtR}}(__doc__XH  Created with lambdify. Signature:

func(arg_0)

Expression:

0.5*I1*q1dot**2 + 0.5*I2*(q1dot + q2dot)**2 + g*m1*r1*cos(q1) +...

Source code:

def _lambdifygenerated(_Dummy_34):
    [m1, l1, r1, I1, m2, l2, r2, I2, g, q1, q2, q1dot, q2dot] = _Dummy_34
    x0 = q1dot**2
    x1 = cos(q1)
    x2 = q1dot + q2dot
    x3 = l1*x1
    x4 = q1 + q2
    x5 = r2*cos(x4)
    x6 = sin(q1)
    x7 = r1**2*x0
    return 0.5*I1*x0 + 0.5*I2*x2**2 + g*m1*r1*x1 + g*m2*(x3 + x5) + 0.5*m1*(x1**2*x7 + x6**2*x7) + 0.5*m2*((q1dot*x3 + x2*x5)**2 + (-l1*q1dot*x6 - r2*x2*sin(x4))**2)


Imported modules:

__annotations__}ubh((cosnumpy.core._multiarray_umathcossinh4sinu0Fh(h(KK K K!KKCB  | \}}}}}}}}}	}
}}}}|d }|| }|d }|| }||d  }t |}|| ||  ||  ||  |d | | |d   ||  d }|| }|| }|t| }|| }|| }|| t |
|  }|| }||	|  |d |  }|| }t |
}|	|| | | ||   ||  |||   } t|g|g|||  ||   g|| |  ||| | | d|     ggS NKGΏπ      h
h	array(	_Dummy_35hhhhhhhhhhhhha2hhhhhhh h!x8x9x10x11x12x13x14x15x16x17t<lambdifygenerated-2>h$KC(  D0))tR}h)Nsh$NNtR}}(h.XJ  Created with lambdify. Signature:

func(arg_0)

Expression:

Matrix([[q1dot], [q2dot], [((I2 + m2*r2**2)*(g*(l1*m2*sin(q1) +...

Source code:

def _lambdifygenerated(_Dummy_35):
    [m1, l1, r1, I1, m2, l2, r2, I2, g, q1, q2, q1dot, q2dot, a2] = _Dummy_35
    x0 = r2**2
    x1 = m2*x0
    x2 = l1**2
    x3 = m2*x2
    x4 = m1*r1**2
    x5 = sin(q2)
    x6 = (I1*I2 + I1*x1 + I2*x3 + I2*x4 + m2**2*x0*x2*x5**2 + x1*x4)**(-1.0)
    x7 = l1*m2
    x8 = r2*x7
    x9 = x8*cos(q2)
    x10 = I2 + x1
    x11 = x10 + x9
    x12 = m2*r2*sin(q1 + q2)
    x13 = x5*x8
    x14 = a2 + g*x12 - q1dot**2*x13
    x15 = q2dot*x13
    x16 = sin(q1)
    x17 = g*(m1*r1*x16 + x12 + x16*x7) + q1dot*x15 + x15*(q1dot + q2dot)
    return array([[q1dot], [q2dot], [x6*(x10*x17 - x11*x14)], [x6*(-x11*x17 + x14*(I1 + x10 + x3 + x4 + 2*x9))]])


Imported modules:

h0}ubhP(arraynumpyarraycosh6sinh9u0Dh(h(KK K K8KKCB6  | \}}}}}}}}}	}
}}}}|d }|| }|d }|| }||d  }t |}|d | | }|| ||  ||  ||  ||  |d |  }|d }t|}|| }|| }|| }|| }|| }| }|
| }|| } | t| }!|	|! }"t|
}#|| }$|	||# |! |#|$   }%| t | }&|d }'|| }(||	|&  |'|(  })||( }*|| }+||( },t |
}-|	||- |$|-  |&  ||*  |+|,  }.d| | | |d  }/|(|) }0|| }1||1 |" |+|1  }2| |' |" }3d|* }4|(|+ |* |, }5|d|  | | | }6|| }7tg d’g d’|||% ||"   |/ ||. ||)   |||2 ||3  |0   |d| | | | | | ||4   ||5 | g|||% |"|6   |/ | |. |)|6   |||2 |(|.  d|0  |3|6    |d| | | | | | |4|6   |5|7 ggtdgdg|7g|6| ggfS (NKGΏπ      (K K KK t(K K K KtK th
h	h=(	_Dummy_36hhhhhhhhhhhhhh@hhhhhhh h!hAhBhChDhEhFhGhHhIhJx18x19x20x21x22x23x24x25x26x27x28x29x30x31x32x33x34x35x36x37x38x39x40t<lambdifygenerated-3>h$KCV  4())tR}h)Nsh$NNtR}}(h.Xγ  Created with lambdify. Signature:

func(arg_0)

Expression:

(Matrix([ [...

Source code:

def _lambdifygenerated(_Dummy_36):
    [m1, l1, r1, I1, m2, l2, r2, I2, g, q1, q2, q1dot, q2dot, a2] = _Dummy_36
    x0 = r2**2
    x1 = m2*x0
    x2 = l1**2
    x3 = m2*x2
    x4 = m1*r1**2
    x5 = sin(q2)
    x6 = m2**2*x0*x2
    x7 = I1*I2 + I1*x1 + I2*x3 + I2*x4 + x1*x4 + x5**2*x6
    x8 = x7**(-1.0)
    x9 = cos(q2)
    x10 = l1*m2
    x11 = r2*x10
    x12 = x11*x9
    x13 = I2 + x1
    x14 = x12 + x13
    x15 = -x14
    x16 = q1 + q2
    x17 = m2*r2
    x18 = x17*cos(x16)
    x19 = g*x18
    x20 = cos(q1)
    x21 = m1*r1
    x22 = g*(x10*x20 + x18 + x20*x21)
    x23 = x17*sin(x16)
    x24 = q1dot**2
    x25 = x11*x5
    x26 = a2 + g*x23 - x24*x25
    x27 = q1dot*x25
    x28 = q1dot + q2dot
    x29 = q2dot*x25
    x30 = sin(q1)
    x31 = g*(x10*x30 + x21*x30 + x23) + q2dot*x27 + x28*x29
    x32 = 2*x5*x6*x9/x7**2
    x33 = x25*x26
    x34 = q2dot*x12
    x35 = q1dot*x34 + x19 + x28*x34
    x36 = -x12*x24 + x19
    x37 = 2*x27
    x38 = x25*x28 + x27 + x29
    x39 = I1 + 2*x12 + x13 + x3 + x4
    x40 = x15*x8
    return (array([[0, 0, 1, 0], [0, 0, 0, 1], [x8*(x13*x22 + x15*x19), -x32*(x13*x31 - x14*x26) + x8*(x13*x35 + x15*x36 + x33), x8*(2*l1*m2*q2dot*r2*x13*x5 - x15*x37), x13*x38*x8], [x8*(x15*x22 + x19*x39), -x32*(-x14*x31 + x26*x39) + x8*(x15*x35 + x25*x31 - 2*x33 + x36*x39), x8*(2*l1*m2*q2dot*r2*x15*x5 - x37*x39), x38*x40]]), array([[0], [0], [x40], [x39*x8]]))


Imported modules:

h0}ubh(arrayh[cosh6sinh9u0Vh(h(KK K KKKCC.| \}}}}}}}}}	}
}}}t |g|ggS Nh=(	_Dummy_37hhhhhhhhhhhhht<lambdifygenerated-4>h$KC ))tR}h)Nsh$NNtR}}(h.X  Created with lambdify. Signature:

func(arg_0)

Expression:

Matrix([[q1dot], [q2dot]])

Source code:

def _lambdifygenerated(_Dummy_37):
    [m1, l1, r1, I1, m2, l2, r2, I2, g, q1, q2, q1dot, q2dot] = _Dummy_37
    return array([[q1dot], [q2dot]])


Imported modules:

h0}ubharrayh[s0u.