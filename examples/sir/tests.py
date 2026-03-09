import sys
import os
import pytest
import numpy as np

SRC_SUBDIR = ''
SRC_SUBDIR = os.path.abspath(SRC_SUBDIR)
if SRC_SUBDIR not in sys.path:
    print(f'Adding source directory to the sys.path: {SRC_SUBDIR!r}')
    sys.path.insert(1, SRC_SUBDIR)

from src.operators.base import check_dot_product, check_linearization
from src.objectives.base import check_objective
from sir_params import SISIZE, SIRMODSIZE
from sir_update import SIRUpdate, SIRSympyUpdate
from sir_modelling import SIRModelling, SIRFixedModelling, SIRSampler, sample
from sir_inversion import sir_objfn


@pytest.mark.parametrize("s, i, a, b", 
    [(0.9, 0.1, 0.21, 0.11),
     (0.999, 0.001, 0.35, 0.11),
     (1.0-1.0e-5, 1.0e-5, 0.3, 0.4)]
)
def test_update(s, i, a, b):
    sirupdate = SIRUpdate(a=a, b=b)
    sympyupdate = SIRSympyUpdate(a=a, b=b)
    v0 = sirupdate._f_si(s, i)
    v1 = sympyupdate._f_si(s, i)
    np.testing.assert_allclose(v0, v1)

@pytest.mark.parametrize("s, i, a, b", 
    [(0.9, 0.1, 0.21, 0.11),
     (0.999, 0.001, 0.35, 0.11),
     (1.0-1.0e-5, 1.0e-5, 0.3, 0.4)]
)
def test_update_jac(s, i, a, b):
    sirupdate = SIRUpdate(a=a, b=b)
    sympyupdate = SIRSympyUpdate(a=a, b=b)
    v0 = sirupdate._jac_si(s, i)
    v1 = sympyupdate._jac_si(s, i)
    np.testing.assert_allclose(v0, v1)
    
@pytest.mark.parametrize("s, i, a, b", 
    [(0.9, 0.1, 0.21, 0.11),
     (0.999, 0.001, 0.35, 0.11),
     (1.0-1.0e-5, 1.0e-5, 0.3, 0.4)]
)
def test_update_jacmod(s, i, a, b):
    sirupdate = SIRUpdate(a=a, b=b)
    sympyupdate = SIRSympyUpdate(a=a, b=b)
    v0 = sirupdate._jac_si(s, i)
    v1 = sympyupdate._jac_si(s, i)
    np.testing.assert_allclose(v0, v1)

def create_sir_mod(sample_i):
    return SIRModelling(T = 100.0, dt =0.1, Tsub =1.0, sample_i=sample_i)

def create_sir_mod_fixed(sample_i):
    m0 = np.array([0.2, 0.1])
    return SIRFixedModelling(T = 100.0, dt =0.1, Tsub =1.0, model0 = m0, sample_i=sample_i)

def create_mod(input_size, name):
    sirmod_size = SISIZE + SIRMODSIZE
    sirmod_sample_i_size = sirmod_size + SIRSampler.ptf_size()
    sirfixedmod_size =  SISIZE
    sirfixedmod_sample_i_size = sirfixedmod_size + SIRSampler.ptf_size()
    if name == "sirmod" and input_size == sirmod_size:
        return create_sir_mod(sample_i = False)
    elif name == "sirmod" and input_size == sirmod_sample_i_size:
        return create_sir_mod(sample_i = True)
    elif name == "sirmodfixed" and input_size == sirfixedmod_size:
        return create_sir_mod_fixed(sample_i = False)
    elif name == "sirmodfixed" and input_size == sirfixedmod_sample_i_size:
        return create_sir_mod_fixed(sample_i = True)
    else:
        raise ValueError(f"{name} and {input_size=} dont match")
    
@pytest.mark.parametrize("input", [
    np.array([0.9, 0.1, 0.21, 0.11]),
])
def test_sample_mod(input):
    ptp = 0.9
    pfp = 0.01
    sirmod = create_mod(input.size, "sirmod")
    data = sirmod(input)
    infectives = data[:,1]
    sir_sample_mod = sirmod.create_sampler()
    predicted = ptp*infectives + pfp*(1-infectives)
    sampler_input = np.concatenate([input, [ptp,pfp]])
    modelled = sir_sample_mod(sampler_input)
    np.testing.assert_equal(modelled, predicted[:,np.newaxis])

@pytest.mark.parametrize("name, input", [
    ("sirmod", np.array([0.9, 0.1, 0.21, 0.11])),
    ("sirmod",np.array([0.9, 0.1, 0.21, 0.11, 0.9, 0.1])),
    ("sirmod", np.array([0.999, 0.001, 0.35, 0.11])),
    ("sirmod",np.array([0.999, 0.001, 0.35, 0.11, 1.0, 0.0])),
    ("sirmod", np.array([1.0-1.0e-5, 1.0e-5, 0.3, 0.4, 0.999, 0.01])),
    ("sirmodfixed", np.array([0.999, 0.001])),
    ("sirmodfixed", np.array([1.0-1.0e-5, 1.0e-5,0.999, 0.01])),
])

def test_dot_product(name, input):
    sirmod = create_mod(input.size, name)
    check_dot_product(sirmod, input)

@pytest.mark.parametrize("name, input, atol", [
    ("sirmod", np.array([0.9, 0.1, 0.21, 0.11, 0.9, 0.1]),1.0e-6),
    ("sirmod", np.array([0.999, 0.001, 0.3, 0.11]),1.0e-4),
    ("sirmod", np.array([0.999, 0.001, 0.3, 0.11, 1.0, 0.0]),1.0e-4),
    ("sirmod", np.array([1.0-1.0e-5, 1.0e-5, 0.3, 0.4, 0.999, 0.001]), 1.0e-6),
    ("sirmodfixed", np.array([0.9, 0.1, 0.9, 0.1]),1.0e-6),
    ("sirmodfixed", np.array([0.998, 0.002, 0.999, 0.001]), 1.0e-6)
])
def test_linearization(name, input, atol):
    sirmod = create_mod(input.size, name)
    check_linearization(sirmod, input, atol=atol)

def create_sir_objfn(input, name):
    sirmod = create_mod(input.size, name = name)
    data = sirmod(input)
    return sir_objfn(data, sirmod)

@pytest.mark.parametrize("name, input", [
    ("sirmod", np.array([0.9, 0.1, 0.21, 0.11, 0.9, 0.1])),
    ("sirmod", np.array([0.999, 0.001, 0.3, 0.11])),
    ("sirmodfixed", np.array([0.9, 0.1, 0.9, 0.1])),
    ("sirmodfixed", np.array([0.998, 0.002, 0.999, 0.001])),
    ("sirmodfixed", np.array([0.999, 0.001])),
])

def test_objective(name, input):
    objective = create_sir_objfn(input, name)
    input = np.zeros((objective.xshape),dtype=np.float64)
    check_objective(objective, input)