import numpy as np
from scipy.optimize import minimize

from models import GeneralizedGDModel, BezierModel
from util import Style


class Optimizer:
    def __init__(self, model, *, param_bounds, init_guess=None, verbose=False):
        self.model = model

        self.param_bounds = param_bounds
        if init_guess is None:
            init_guess = [np.mean(bounds) for bounds in param_bounds]
        self.init_guess = np.array(init_guess)

        self.verbose = bool(verbose)
        self.print = print if verbose else (lambda *_, **__: None)

        self.opt_params = None

    def fit(self, points):
        points = np.array(points, copy=True)

        # The function we want to minimize; given params, we set them as model params then compute and return the MSE
        def objective_function(params):
            self._set_model_params(params)
            return self.mse(points)

        # Run L-BFGS-B to find parameters that optimally fit the model to the given points
        result = minimize(objective_function, self.init_guess, method='L-BFGS-B', bounds=self.param_bounds)
        self.opt_params = result.x

        # Ensure model is configured with optimal parameters
        self._set_model_params(self.opt_params)

        # Performance metrics
        mse = self.mse(points)
        mae = self.mae(points)
        R2 = self.R2(points)

        # [Verbose] Show optimal parameters, optimized model, performance metrics, and plot model curve against points
        col_width = max(map(len, self.model.PARAM_NAMES))
        self.print()
        self.print('-' * 75)
        self.print(Style('OPTIMIZER RESULTS').bold())
        self.print()
        self.print(Style('Optimal Parameters:').blue())
        self.print(*(f'{name.ljust(col_width)} = {value}' for name, value in self.model.params.items()), sep='\n')
        self.print()
        self.print(Style('Optimized Model:').blue())
        self.print(self.model)
        self.print()
        self.print(Style('Performance Metrics:').green())
        self.print(f'MSE:  {mse}')
        self.print(f'MAE:  {mae}')
        self.print(f'R^2:  {R2}')
        self.print('-' * 75)
        self.print()
        if self.verbose:
            self.model.plot(points=points)

        return self.model

    def mse(self, points):
        x, y = points.T
        y_pred = self.model(x)
        return np.mean((y - y_pred) ** 2)

    def mae(self, points):
        x, y = points.T
        y_pred = self.model(x)
        return np.mean(np.abs(y - y_pred))

    def R2(self, points):
        x, y = points.T
        y_pred = self.model(x)
        residual_sum_of_squares = np.sum((y - y_pred) ** 2)
        total_sum_of_squares = np.sum((y - np.mean(y)) ** 2)
        return 1 - (residual_sum_of_squares / total_sum_of_squares)

    def _set_model_params(self, params):
        self.model.set_params(**{name: value for name, value in zip(self.model.PARAM_NAMES, params)})


def main():
    # Generate points using GeneralizedGDModel, then fit a BezierModel to them

    def noise(delta, size):
        return np.random.uniform(-delta, delta, size)

    # Generate data points we want to fit to
    target_model = GeneralizedGDModel(H=500, Z=60, alpha=0.4, beta=0.9, k=0.75)
    x = np.linspace(0, target_model.H, 100)
    y = target_model(x) + noise(5, len(x))
    points = np.array(list(zip(x, y)))

    # Instantiate BezierModel that we want to optimize
    degree = 3
    inner_control_points = [(i, 0) for i in np.linspace(0, 1, degree)]
    model = BezierModel(inner_control_points, H=500, Z=60)

    # Define param bounds for the BezierModel (must do this carefully as x must be strictly increasing for CubicSpline)
    left_skew = 3  # generally, we get better accuracy with denser control points near x=0 (steepest part of bowl)
    x_bounds = [((i/degree)**left_skew, ((i+1)/degree)**left_skew) for i in range(degree)]
    y_bounds = [(0, 1)] * degree
    param_bounds = [val for pair in zip(x_bounds, y_bounds) for val in pair]  # interleave x and y bounds

    # See model curve before fitting
    model.plot(points=points)

    # Instantiate optimizer and fit it to our generated data points
    optimizer = Optimizer(
        model,
        param_bounds=param_bounds,
        verbose=True
    )
    optimizer.fit(points)


if __name__ == '__main__':
    main()
