import inspect
from collections.abc import Callable
from typing import SupportsIndex

import numpy as np
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL, ExtendedUnbinnedNLL
from IPython.display import display

from IPython.display import Latex, display
from sympy import *
import matplotlib.pyplot as plt

def insert(vector) -> list:
    """
    Generate symbolic variables, sigmas, covariances, and a list of all error terms for a given vector.

    Parameters
    ----------
    vector : list of str
        List of variable names as strings.

    Returns
    -------
    vector : sympy.Symbol or tuple of sympy.Symbol
        The symbolic variables corresponding to the input names.
    sigmas : sympy.Symbol or tuple of sympy.Symbol
        The sigma (uncertainty) symbols for each variable.
    covar : sympy.Symbol or tuple of sympy.Symbol or None
        The covariance symbols between variables, or None if not applicable.
    all : list of str
        List of all sigma and covariance symbol names as strings.
    """
    # Handling edge cases
    if len(vector) == 1:
        vector = symbols(str(vector))
        sigmas = f"sigma_{vector}"
        covar = None
        all = sigmas
    elif len(vector) == 2:
        string = ' '.join(vector)
        vector = symbols(string)
        # Creation of sigmas and covariants
        sigmas = []
        covar = []
        all = []
        for i in range(len(vector)):
            sigmas.append(f"sigma_{vector[i]}")
            all.append(f"sigma_{vector[i]}")
        sigmastring = ' '.join(sigmas)
        sigmas = symbols(sigmastring)
        covar.append(f"sigma_{vector[0]}{vector[1]}")
        covarstring = ' '.join(covar)
        covar = symbols(covarstring)
        all.append(covar)
    else:
        string = ' '.join(vector)
        vector = symbols(string)
        # Creation of sigmas and covariants
        sigmas = []
        covar = []
        all = []
        for i in range(len(vector)):
            for j in range(i, len(vector)):
                if i == j:
                    sigmas.append(f"sigma_{vector[i]}")
                    all.append(f"sigma_{vector[i]}")
                else:
                    covar.append(f"sigma_{vector[i]}{vector[j]}")
                    all.append(f"sigma_{vector[i]}{vector[j]}")
        sigmastring = ' '.join(sigmas)
        sigmas = symbols(sigmastring)
        covarstring = ' '.join(covar)
        covar = symbols(covarstring)
    return vector, sigmas, covar, all

def derivazione(variables, formula, sigmas, covar) -> str:
    """
    Compute the symbolic error propagation formula using derivatives.

    Parameters
    ----------
    variables : tuple of sympy.Symbol
        The variables in the formula.
    formula : sympy expression
        The formula for which to propagate errors.
    sigmas : tuple of sympy.Symbol
        The sigma (uncertainty) symbols for each variable.
    covar : tuple of sympy.Symbol or sympy.Symbol or None
        The covariance symbols between variables.

    Returns
    -------
    expo : sympy expression
        The symbolic expression for the error propagation.
    """
    expo = 0
    # Add the sigmas
    for i in range(len(variables)):
        expo += (diff(formula, variables[i]))**2 * sigmas[i]
    # Add the covariances
    k = 0
    if isinstance(covar, Symbol):
        expo += 2 * (diff(formula, variables[0])) * (diff(formula, variables[1])) * covar
    else:
        for i in range(len(variables)):
            for j in range(len(variables) - i - 1):
                expo += 2 * (diff(formula, variables[i])) * (diff(formula, variables[1 + j + i])) * covar[k]
                k += 1
    return expo

def propagazione_errore(variables, formula, values, covmat, additional_variables=None, additional_variable_values=None, Display=True) -> float:
    """
    Perform error propagation for a given formula and variables.

    Parameters
    ----------
    variables : list of str
        List of variable names as strings.
    formula : str or sympy expression
        The formula for which to propagate errors.
    values : list
        List of values for each variable.
    covmat : list of list
        Covariance matrix (or list of variances and covariances).
    additional_variables : list of str, optional
        Additional variables to substitute.
    additional_variable_values : list, optional
        Values for the additional variables.
    Display : bool, optional
        Whether to display the result as LaTeX.

    Returns
    -------
    result : sympy expression
        The propagated error as a symbolic expression.
    """
    if isinstance(additional_variables, list):
        things, _ , _ , _ = insert(additional_variables)

    variables, sigmas, covar, all = insert(variables)
    expo = derivazione(variables, formula, sigmas, covar)

    if Display:
        a = '\sigma=' + latex(simplify(sqrt(expo)))
        ax = plt.axes([0,0,0.5,0.5]) #left,bottom,width,height
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        plt.text(0,1,'$%s$' %a,size=50,color="green")
        plt.show()

    if len(variables) == 1:
        expo = expo.subs(variables, values)
        expo = expo.subs(sigmas, covmat)
    else:
        for i in range(len(variables)):
            expo = expo.subs(variables[i], values[i])
        k = 0
        for i in range(len(covmat)):
            expo = expo.subs(all[k], covmat[i][i])
            k += 1
        for i in range(len(covmat)):
            for j in range(i + 1, len(covmat[i])):
                expo = expo.subs(all[k], covmat[i][j])
                k += 1
    if isinstance(additional_variables, list):
        if len(additional_variables) == 1:
            expo = expo.subs(symbols(additional_variables[0]), additional_variable_values[0])
        else:
            for i in range(len(additional_variables)):
                expo = expo.subs(things[i], additional_variable_values[i])
    return sqrt(expo)