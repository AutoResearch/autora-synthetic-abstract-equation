import warnings
from functools import partial
from itertools import product
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sympy import Expr, lambdify

from autora.experiment_runner.synthetic.utilities import SyntheticExperimentCollection
from autora.variable import DV, IV, VariableCollection


def equation_experiment(
    expression: Expr,
    X: Optional[List[IV]] = None,
    y: Optional[DV] = None,
    name: str = "Equation Experiment",
    added_noise: float = 0.001,
    random_state: int = 42,
    rename_output_columns: bool = True,
):
    """

    A synthetic experiments that uses a sympy expression as ground truth.

    Sympy: https://www.sympy.org/en/index.html

    Args:
        expression: A sympy expression. The expression is interpreted as definition for a function
        X: The domain of independent variables
        y: The codomain of the dependent variables
        name: Name of the experimen
        added_noise: Standard deviation of gaussian noise added to output
        random_state: Seed for random number generator
        rename_output_columns: If true, rename the columns of the output DataFrame based on the
            variable names in the expression.


    Examples:
        First we define an expression that will be interpreted as function. We need to define the
        symbols in sympy.
        >>> from sympy import symbols
        >>> x, y = symbols("x y")

        Now we can define an expression:
        >>> expr = x ** y

        Then we use this expression in our experiment
        >>> experiment = equation_experiment(expr)

        To run an experiment, we can either use a numpy array:
        >>> experiment.experiment_runner(np.array([[1, 1], [2 ,2], [2 ,3]]))
                  x         y  observations
        0  1.000305  0.998960      1.000304
        1  2.000750  2.000941      4.005614
        2  1.998049  2.998698      7.969424

        But we have to be carefull with the order of the arguments in the runner. The arguments
        will be sorted alphabetically.
        In the following case the first entry of the numpy array is still x:
        >>> expr = y ** x
        >>> experiment = equation_experiment(expr)
        >>> experiment.experiment_runner(np.array([[1, 1], [2, 2] , [2, 3]]))
                  x         y  observations
        0  1.000305  0.998960      0.998960
        1  2.000750  2.000941      4.005848
        2  1.998049  2.998698      8.972943

        We can be more secure by using Pandas Dataframes as an input:
        >>> experiment.experiment_runner(pd.DataFrame({'y':[1, 2, 2], 'x': [1, 2, 3]}))
                  y         x  observations
        0  0.999684  1.000128      0.999684
        1  1.999147  1.999983      3.996542
        2  2.000778  3.000879      8.014223

        If we use columns in the dataframe, that are not part of the expression,
        we will get an error message:
        >>> experiment.experiment_runner(pd.DataFrame({'z':[1, 2, 2], 'x': [1, 2, 3]}))
        Traceback (most recent call last):
        ...
        Exception: Variables of expression y**x not found in columns of dataframe with columns\
 Index(['z', 'x'], dtype='object')

        If we want to use additional functionality of the synthetic model like plotting, we need
        to define a domain:
        >>> variable_x = IV(
        ... name="x",
        ... allowed_values=np.linspace(-10, 10, 100),
        ... value_range=(-10, 10),
        ... )
        >>> variable_y = IV(
        ... name="y",
        ... allowed_values=np.linspace(-10, 10, 100),
        ... value_range=(-10, 10),
        ... )

        Reinitialise the experiment with the domain and printing it
        >>> expr = x ** 2 + y ** 2
        >>> experiment = equation_experiment(expr, X = [variable_x, variable_y])
        >>> experiment.domain() # doctest: +ELLIPSIS
        array([[-10.       , -10.       ],
               ...
               [ 10.       ,  10.       ]])

        You can plot the experiment with
        >>> experiment.plotter()
    """

    params = dict(
        # Include all parameters here:
        expression=expression,
        name=name,
        added_noise=added_noise,
        random_state=random_state,
    )

    args = list(expression.free_symbols)
    args = sorted(args, key=lambda el: el.name)

    f_numpy = lambdify(args, expression, "numpy")

    # Define variables
    variables = VariableCollection(
        independent_variables=[X],
        dependent_variables=[y],
    )

    # Define experiment runner
    rng = np.random.default_rng(random_state)

    def experiment_runner(
        x: Union[pd.DataFrame, np.ndarray, np.recarray], added_noise_=added_noise
    ):
        """A function which simulates noisy observations."""
        if isinstance(x, pd.DataFrame):
            if not set([el.name for el in args]).issubset(x.columns):
                raise Exception(
                    f"Variables of expression {expression} "
                    f"not found in columns of dataframe with columns {x.columns}"
                )
            x_filtered = x[[el.name for el in args]]
            x_sorted = x_filtered.sort_index(axis=1)
            x_ = np.array(x_sorted)
        else:
            x_ = x
            warnings.warn(
                "Unnamed data is used. Arguments will be sorted alphabetically. "
                "Consider using a Pandas DataFrame with named columns for "
                "better clarity and ease of use.",
                category=RuntimeWarning,
            )

        x_ = x_ + rng.normal(0, added_noise_, size=x_.shape)

        y = f_numpy(*x_.T)
        if isinstance(x, pd.DataFrame):
            _res = pd.DataFrame(x_, columns=x_sorted.columns)
            res = x
            for col in x_sorted.columns:
                res[col] = _res[col]
        else:
            if rename_output_columns:
                res = pd.DataFrame(x_, columns=[el.name for el in args])
            else:
                res = pd.DataFrame(x_.T)

        res["observations"] = y
        return res

    ground_truth = partial(experiment_runner, added_noise_=0.0)
    """A function which simulates perfect observations"""

    def domain():
        """A function which returns all possible independent variable values as a grid."""
        iv_values = [iv.allowed_values for iv in variables.independent_variables[0]]
        X_combinations = product(*iv_values)
        X = np.array(list(X_combinations))
        return X

    def plotter(model=None):
        """A function which plots the ground truth and (optionally) a fitted model."""
        import matplotlib.pyplot as plt

        plt.figure()
        dom = domain()
        data = ground_truth(dom)

        y = data["observations"]
        x = data.drop("observations", axis=1)

        if x.shape[1] > 2:
            Exception(
                "No standard way to plot more then 2 independent variables implemented"
            )

        if x.shape[1] == 1:
            plt.plot(x, y, label="Ground Truth")
            if model is not None:
                plt.plot(x, model.predict(x), label="Fitted Model")
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            x_ = x.iloc[:, 0]

            y_ = x.iloc[:, 1]
            z_ = y

            ax.scatter(x_, y_, z_, s=1, alpha=0.3, label="Ground Truth")
            if model is not None:
                z_m = model.predict(x)
                ax.scatter(x_, y_, z_m, s=1, alpha=0.5, label="Fitted Model")

        plt.legend()
        plt.title(name)
        plt.show()

    # The object which gets stored in the synthetic inventory
    collection = SyntheticExperimentCollection(
        name=name,
        description=equation_experiment.__doc__,
        variables=variables,
        experiment_runner=experiment_runner,
        ground_truth=ground_truth,
        domain=domain,
        plotter=plotter,
        params=params,
        factory_function=equation_experiment,
    )
    return collection

