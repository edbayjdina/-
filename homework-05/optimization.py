import numpy as np
from numpy.linalg import LinAlgError
import scipy
from scipy.linalg import cho_factor, cho_solve
from datetime import datetime
from collections import defaultdict

try:
    from scipy.optimize.linesearch import scalar_search_wolfe2
except Exception:
    from scipy.optimize._linesearch import scalar_search_wolfe2


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        if self._method == 'Constant':
            return self.c

        alpha_0 = self.alpha_0 if previous_alpha is None else previous_alpha

        def armijo_backtracking(start_alpha):
            alpha = start_alpha
            phi_0 = oracle.func_directional(x_k, d_k, 0.0)
            derphi_0 = oracle.grad_directional(x_k, d_k, 0.0)
            if not (np.isfinite(phi_0) and np.isfinite(derphi_0)):
                return None
            while alpha > 0:
                phi_alpha = oracle.func_directional(x_k, d_k, alpha)
                if not np.isfinite(phi_alpha):
                    return None
                if phi_alpha <= phi_0 + self.c1 * alpha * derphi_0:
                    return alpha
                alpha *= 0.5
            return None

        if self._method == 'Armijo':
            return armijo_backtracking(alpha_0)

        if self._method == 'Wolfe':
            try:
                if alpha_0 <= 0:
                    alpha_0 = self.alpha_0
                phi = lambda t: oracle.func_directional(x_k, d_k, alpha_0 * t)
                derphi = lambda t: alpha_0 * oracle.grad_directional(x_k, d_k, alpha_0 * t)
                alpha_rel, _, _, _ = scalar_search_wolfe2(phi, derphi, c1=self.c1, c2=self.c2)
                if alpha_rel is not None and np.isfinite(alpha_rel):
                    alpha = alpha_0 * alpha_rel
                    if np.isfinite(alpha) and alpha > 0:
                        return alpha
            except Exception:
                pass
            return armijo_backtracking(alpha_0)

        raise ValueError('Unknown method {}'.format(self._method))



def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()



def _append_history(history, start_time, x_k, func_value, grad_value):
    history['time'].append((datetime.now() - start_time).total_seconds())
    history['func'].append(func_value)
    history['grad_norm'].append(np.linalg.norm(grad_value))
    if x_k.size <= 2:
        history['x'].append(np.copy(x_k))



def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    start_time = datetime.now()
    previous_alpha = None

    for _ in range(max_iter):
        f_k = oracle.func(x_k)
        g_k = oracle.grad(x_k)
        if not (np.all(np.isfinite(np.asarray(f_k))) and np.all(np.isfinite(np.asarray(g_k)))):
            return x_k, 'computational_error', history

        if trace:
            _append_history(history, start_time, x_k, f_k, g_k)

        if np.linalg.norm(g_k) <= tolerance:
            return x_k, 'success', history

        d_k = -g_k
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=previous_alpha)
        if alpha is None or not np.isfinite(alpha) or alpha <= 0:
            return x_k, 'computational_error', history

        x_k = x_k + alpha * d_k
        previous_alpha = alpha

        if display:
            print('f = {:.6e}, ||g|| = {:.6e}, alpha = {:.6e}'.format(f_k, np.linalg.norm(g_k), alpha))

    return x_k, 'iterations_exceeded', history



def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    start_time = datetime.now()
    previous_alpha = None

    for _ in range(max_iter):
        f_k = oracle.func(x_k)
        g_k = oracle.grad(x_k)
        H_k = oracle.hess(x_k)
        if not (np.all(np.isfinite(np.asarray(f_k))) and np.all(np.isfinite(np.asarray(g_k))) and np.all(np.isfinite(np.asarray(H_k)))):
            return x_k, 'computational_error', history

        if trace:
            _append_history(history, start_time, x_k, f_k, g_k)

        if np.linalg.norm(g_k) <= tolerance:
            return x_k, 'success', history

        try:
            c, lower = cho_factor(H_k)
            d_k = -cho_solve((c, lower), g_k)
        except LinAlgError:
            return x_k, 'newton_direction_error', history

        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=previous_alpha)
        if alpha is None or not np.isfinite(alpha) or alpha <= 0:
            return x_k, 'computational_error', history

        x_k = x_k + alpha * d_k
        previous_alpha = alpha

        if display:
            print('f = {:.6e}, ||g|| = {:.6e}, alpha = {:.6e}'.format(f_k, np.linalg.norm(g_k), alpha))

    return x_k, 'iterations_exceeded', history
