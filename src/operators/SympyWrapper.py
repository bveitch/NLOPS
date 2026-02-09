from sympy import symbols, Matrix, sympify, lambdify, expand, simplify
import numpy as np
import numpy.typing as npt
 
class SympyWrap:
 
    def __init__(self, expression):
        self._expression = expression
        self._is_matrix = True
        self._shape = None
        try:
            self._shape = expression.shape
        except AttributeError:
            self._is_matrix = False
        self._symbols = self._expression.free_symbols
        self._symbol_map = {symbol.name : symbol for symbol in self._symbols}
 
    @classmethod  
    def from_string(cls, expr: str):
        return cls(sympify(expr))
   
    @classmethod
    def from_strings(cls, exprs: list[str]):
        return cls(Matrix(exprs))
   
    def __str__(self):
        return str(self.expression)
   
    @property
    def expression(self):
        return self._expression
 
    @property
    def symbols(self):
        return self._symbols
   
    @property
    def nsymbols(self):
        return len(self._symbols)
   
    @property
    def symbol_names(self):
        return set(self._symbol_map.keys())
   
    @property 
    def is_vector(self):
        if self._shape is not None:
            return self._shape[1] == 1
        return False
    
    def get_symbol(self, symbol_name: str):
        if symbol_name in self._symbol_map:
            return self._symbol_map[symbol_name]
        raise KeyError(f"{symbol_name} not found in {self.symbol_names}")
   
    def get_subs(self, symbol_dict: dict[str,float]):
        subs = {}
        for symbol_name, value in symbol_dict.items():
            try:
                subs[self.get_symbol(symbol_name)] = value
            except KeyError:
                pass
        return subs
   
    def partial_eval(self, partial_values: dict[str,float], check_keys=False):
        if check_keys:
            assert set(partial_values.keys()).issubset(self.symbol_names), "keys dont match symbols"
        partial_subs = self.get_subs(partial_values)
        partial_expression = self._expression.evalf(subs=partial_subs)
        return SympyWrap(partial_expression)
   
    def jac(self, symbol_names:list[str]):
        ordered_symbols = [self._symbol_map[symbol_name] for symbol_name in symbol_names]
        if self._is_matrix:
            matrix= Matrix([self._expression.diff(s).T for s in ordered_symbols]).T
            return SympyWrap(matrix)
        else:    
            matrix = Matrix([self._expression.diff(s) for s in ordered_symbols])
            return SympyWrap(matrix)
   
    def partial_jac(self, partial_values: dict[str:float]):
        symbol_names = partial_values.keys()
        jac = self.jac(symbol_names)
        return jac.partial_eval(partial_values), symbol_names
   
    def __call__(self, x : npt.NDArray, symbol_names : list[str], axis = 0 ):
        assert len(symbol_names) == self.nsymbols, f"{len(symbol_names)} does not match {self.nsymbols}"
        assert x.shape[axis] == self.nsymbols, f"xshape[{axis}] = {x.shape[axis]} does not match {self.nsymbols}"
        ordered_symbols = [self._symbol_map[symbol_name] for symbol_name in symbol_names]
        f = lambdify(ordered_symbols, self._expression, "numpy")
        xarr = np.unstack(x,axis=axis)
        v = np.array(f(*xarr), dtype = x.dtype)
        if self.is_vector:
            return np.squeeze(v,axis=1)
        return v
   
    def check(self, expression):
       test = simplify(expand(self.expression - expression))
       if self._is_matrix:
           return test.is_zero_matrix
       else:
           return test.is_zero
 
def rosenbrock():
    expr = "(a-x)**2+b*(y-x**2)**2"
    f = sympify(expr)
    df_xy = Matrix(["-2*(a-x)-4*b*x*(y-x**2)", "2*b*(y-x**2)"])
    d2f_xy = Matrix([
        ["2 -4*b*y +4*b*x**2 +8*b*x**2", "-4*b*x"],
        ["-4*b*x", "2*b"]])
    return expr, f, df_xy, d2f_xy

def sir():
    exprs = ["-a*s*i", "a*s*i - b*i", "b*i"]
    f = Matrix(exprs)
    df_dab = Matrix([
        ["-s*i",  "0"],
        [ "s*i", "-i"],
        [  "0",  "i"]
    ])
 
    df_dsir = Matrix([
        ["-a*i", "-a*s"  ],
        [ "a*i",  "a*s-b"],
        [   "0",      "b"]
    ])
    return exprs, f, df_dab, df_dsir
 
def xyz():
    exprs = ["x*y*z", "x+y+z"]
    f = Matrix(exprs)
    df_dxyz = Matrix([
        ["y*z",  "x*z", "x*y"],
        [  "1",    "1",   "1"]
    ])
    return exprs, f, df_dxyz
 
def example_rosenbrock():
    expr, f, df, d2f = rosenbrock()
    symop = SympyWrap.from_string(expr)
    rosenb = symop.partial_eval({"a":1, "b":100})
    subs = symop.get_subs({"a":1, "b":100})
    print(rosenb)
    print(subs)
    f_at_ab = f.evalf(subs=subs)
    assert rosenb.check(f_at_ab), "value incorrect"
    print(rosenb.symbols)
    print(rosenb.symbol_names)
    x= np.array([1,3])
    vars = ['x', 'y']
    v = rosenb(x, vars)
    print(f"f({x}) = {v}")
    jac = rosenb.jac(vars)
    print(jac)
    df_at_ab = df.evalf(subs=subs)
    print(jac)
    assert jac.check(df_at_ab), "jacobian incorrect"
    jx = jac(x,vars)
    print(f"jac({x}) = {jx})")
    hess = jac.jac(vars)
    print(hess)
    d2f_at_ab = d2f.evalf(subs=subs)
    assert hess.check(d2f_at_ab), "hessian incorrect"
 
def example_xyz():
    exprs, f, df_dxyz = xyz()
    symop = SympyWrap.from_strings(exprs)
    assert symop.check(f), "value incorrect"
    x = np.array([1,2,0.1])
    vars = ['x', 'y', 'z']
    v = symop(x, vars)
    print(f"f({x}) = {v}")
    subs = symop.get_subs({var: x[i] for i, var in enumerate(vars)})
    f_at_x = np.array(f.evalf(subs=subs))
    np.testing.assert_array_equal(v, f_at_x)
    jac =symop.jac(vars)
    assert jac.check(df_dxyz), "jacobian incorrect"
    jacv = jac(x,vars)
    df_dxyz_at_x = np.array(df_dxyz.evalf(subs=subs))
    np.testing.assert_array_equal(jacv, df_dxyz_at_x)
 
def example_sir():
    exprs =  ["-a*s*i", "a*s*i-b*i", "b*i"]
    exprs, f, df_dab, df_dsir = sir()
    baseop = SympyWrap.from_strings(exprs)
 
    sirop = baseop.partial_eval({"a":0.1, "b":1.2})
    subs_ab = baseop.get_subs({"a":0.1, "b":1.2})
    f_at_ab = f.evalf(subs=subs_ab)
    print(sirop)
    assert sirop.check(f_at_ab), "value incorrect"
    x = np.array([1.0, 1.0])
    vars = ['s', 'i']
    v = sirop(x, vars)
 
    print(f"f({x}) = {v}")
    subs_si = sirop.get_subs({var: x[i] for i, var in enumerate(vars)})  
    f_at_si = np.array(f_at_ab.evalf(subs=subs_si), dtype=v.dtype)
    np.testing.assert_array_equal(v, f_at_si)
 
    jac =sirop.jac(vars)
    df_dsir_at_ab = df_dsir.evalf(subs=subs_ab)
    assert jac.check(df_dsir_at_ab), "jacobian sir incorrect"
    jacsi = jac(x,vars)
    df_dsir_at_si = np.array(df_dsir_at_ab.evalf(subs=subs_si),dtype=jacsi.dtype)
    np.testing.assert_array_equal(jacsi, df_dsir_at_si)
    print(f"jac({x}) = {jac}")
    jac_ab, ab_symbol_names = baseop.partial_jac({"a":0.1, "b":1.2})
    df_dab_at_ab = df_dab.evalf(subs=subs_ab)
    assert jac_ab.check(df_dab_at_ab), "jacobian ab incorrect"
    jac_ab_si = jac_ab(x,vars)
    df_dab_at_si = np.array(df_dab_at_ab.evalf(subs=subs_si), dtype=jac_ab_si.dtype)
    np.testing.assert_array_equal(jac_ab_si, df_dab_at_si)
 
if __name__ == "__main__":
    example_rosenbrock()
    example_xyz()
    example_sir()
 