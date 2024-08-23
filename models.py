import math
import textwrap
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

from util import FixedDict


# Abstract base class for all models
class OudBowlProfileModel(ABC):
    EPSILON = 1e-100  # we treat 0 as EPSILON to avoid bad behavior in some operations (e.g. derivative, integration)
    NECK_LENGTH = 200  # standard neck length in mm; this is fixed across virtually all standard ouds, +/- 5 mm

    def __init__(self, *, H, Z, **params):
        self.H = H
        self.Z = Z
        self.params = FixedDict(params)

        self._check_class_properties()
        self._check_num_params(len(self.params))
        self._check_param_names(*self.params.keys())

        # Subclasses can also choose to do this explicitly for clarity and IDE functionality, but it is set
        # automatically anyway
        self._set_params_as_attributes()

    @property
    @abstractmethod
    def PARAM_NAMES(self):
        # Source of truth for param names and count
        pass

    @abstractmethod
    def psi(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass

    def get_param(self, name):
        self._check_param_names(name)
        return self.params[name]

    def set_param(self, name, value):
        self._check_param_names(name)
        self.params[name] = value
        setattr(self, name, value)

    def get_params(self):
        return self.params.copy()

    def set_params(self, **new_params):
        self._check_param_names(*new_params.keys())
        for key, value in new_params.items():
            # FixedDict has no `update` method, to prevent changing its size
            self.params[key] = value
        self._set_params_as_attributes()

    def plot(self, *, points=None, neck=False, color='dodgerblue', show=True, save_to=None, ax=None, title=None):
        x = np.linspace(0, self.H, 100000)
        y = self.psi(x)

        neck_thickness = self.Z / 2

        # Create figure and axes if not provided
        if ax is None:
            _fig, ax = plt.subplots()

        # Plot options (order matters)
        if points is not None:
            points = np.array(points)
            ax.plot(points[:, 0], points[:, 1], '.', color='black')
        if neck:
            ax.plot([self.H, self.H + self.NECK_LENGTH], [neck_thickness] * 2, color=color)

        # Plot model curve
        ax.plot(x, y, color=color)

        # Set x bounds
        x_max = 1.1 * (self.H + (self.NECK_LENGTH * neck))
        x_min = -0.05 * x_max
        ax.set_xlim(x_min, x_max)

        # Set y bounds
        y_max = 1.5 * (max(max(y), neck_thickness) if neck else max(y))
        ax.set_ylim(0, y_max)

        # Set extra settings and display graph
        if title is None:
            # Model representation (wrapped)
            title = textwrap.fill(str(self), width=60)
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)

        # Must save fig BEFORE showing, or else will save blank image
        if save_to is not None:
            plt.savefig(save_to, dpi=600)
        if show:
            plt.show()

        return ax

    def __call__(self, x):
        return self.psi(x)

    def __repr__(self):
        params_str = ''.join(f', {name}={value}' for name, value in self.params.items())
        return f'{self.__class__.__name__}(H={self.H}, Z={self.Z}{params_str})'

    def _check_class_properties(self):
        required_class_properties = [
            'PARAM_NAMES'
        ]
        for prop_name in required_class_properties:
            if not isinstance(getattr(type(self), prop_name, None), property):
                raise TypeError(f'subclass of {OudBowlProfileModel.__name__} must define {prop_name} as a property (use @property decorator)')

    def _check_num_params(self, num_params):
        if num_params != len(self.PARAM_NAMES):
            raise ValueError(f'unexpected number of params; expected {len(self.params)}, got {num_params}')

    def _check_param_names(self, *names):
        truth_set = set(self.PARAM_NAMES)
        for name in names:
            if name not in truth_set:
                if len(self.PARAM_NAMES) == 1:
                    raise ValueError(f'unexpected param name; expected "{self.PARAM_NAMES[0]}", got "{name}"')
                else:
                    raise ValueError(f'unexpected param name; expected one of {self.PARAM_NAMES}, got "{name}"')

    def _sanitize_singularities(self, singularities):
        H = self.H
        EPSILON = OudBowlProfileModel.EPSILON

        # Modifies in-place
        # NumPy array-friendly
        if isinstance(singularities[0], np.ndarray):
            for s in singularities:
                s[s == 0] = EPSILON
                s[s == H] = H - EPSILON
        else:
            for i in range(len(singularities)):
                if singularities[i] == 0:
                    singularities[i] = EPSILON
                elif singularities[i] == H:
                    singularities[i] = H - EPSILON

    # Optionally used by subclasses if desired, e.g. for arbitrary param count
    def _set_params_as_attributes(self):
        for key, value in self.params.items():
            setattr(self, key, value)


class StaticModel(OudBowlProfileModel):
    # Parameterless model
    def __init__(self, *, H, Z):
        super().__init__(H=H, Z=Z)

    @property
    def PARAM_NAMES(self):
        return tuple()

    def psi(self, x):
        H = self.H
        Z = self.Z
        return H * (np.sqrt(x/H) - 2**(x/H) + 1) + (Z / (2*H))*x

    def derivative(self, x, sanitize_singularities=True):
        H = self.H
        Z = self.Z

        singularities = [x/H]
        if sanitize_singularities:
            self._sanitize_singularities(singularities)

        s = singularities[0]

        return 1/(2*np.sqrt(s)) - 2**(x/H)*math.log(2) + Z/(2*H)


class GrowthDecayModel(OudBowlProfileModel):
    def __init__(self, *, H, Z, alpha):
        super().__init__(H=H, Z=Z, alpha=alpha)
        self.alpha = alpha

    @property
    def PARAM_NAMES(self):
        return ('alpha',)

    def psi(self, x):
        H = self.H
        Z = self.Z
        alpha = self.alpha
        return H * ((x/H)**alpha - 2**(x/H) + 1) + (Z / (2*H))*x

    def derivative(self, x, sanitize_singularities=True):
        H = self.H
        Z = self.Z
        alpha = self.alpha

        singularities = [x/H]
        if sanitize_singularities:
            self._sanitize_singularities(singularities)

        s = singularities[0]

        return alpha * s**(alpha-1) - (math.log(2) * 2**(x/H)) + Z/(2*H)


class GeneralizedGDModel(OudBowlProfileModel):
    """
        The advantage of this model over the BezierModel is that it is noise-resistant when fitting to points, meaning
        it will maintain a proper oud shape even when fitting to highly irregular points. But with this advantage comes
        a lower level of flexibility, although the extent of this disadvantage may or may not be negligible in reality.
    """
    def __init__(self, *, H, Z, alpha, beta, k):
        super().__init__(H=H, Z=Z, alpha=alpha, beta=beta, k=k)
        self.alpha = alpha
        self.beta = beta
        self.k = k

    @property
    def PARAM_NAMES(self):
        return ('alpha', 'beta', 'k')

    def psi(self, x):
        H = self.H
        Z = self.Z
        alpha = self.alpha
        beta = self.beta
        k = self.k
        return H*k * ((x/H)**alpha - 2**(x/H) + 1)**beta + (Z / (2*H))*x

    def derivative(self, x, sanitize_singularities=True):
        H = self.H
        Z = self.Z
        alpha = self.alpha
        beta = self.beta
        k = self.k

        singularities = [
            (x/H)**alpha - 2**(x/H) + 1,
            x/H
        ]
        if sanitize_singularities:
            self._sanitize_singularities(singularities)

        s1, s2 = singularities

        return k * beta * s1**(beta - 1)  \
            * (alpha * s2**(alpha-1) - (math.log(2) * 2**(x/H)))  \
            + Z / (2*H)

    @classmethod
    def slim_preset(cls, *, H=500, Z=60):
        return cls(
            H=H,
            Z=Z,
            alpha=0.456,
            beta=0.916,
            k=0.858
        )

    @classmethod
    def medium_preset(cls, *, H=500, Z=60):
        return cls(
            H=H,
            Z=Z,
            alpha=0.421,
            beta=0.935,
            k=0.887
        )

    @classmethod
    def wide_preset(cls, *, H=500, Z=60):
        return cls(
            H=H,
            Z=Z,
            alpha=0.427,
            beta=0.853,
            k=0.885
        )


class BezierModel(OudBowlProfileModel):
    """
        The advantage of this model over the GeneralizedGDModel is that it is extremely flexible and thus can very
        accurately fit many curve shapes, but with this flexibility comes the issue of overfitting when the points have
        a high degree of noise. This means that when fitting to highly irregular points, this model may not maintain
        a proper oud shape.
    """
    def __init__(self, inner_control_points=None, *, H, Z, **coords_by_name):
        # Ensure exactly one input method is being used
        if (inner_control_points is not None) and (len(coords_by_name) > 0):
            raise ValueError('cannot provide both inner_control_points and individual control point keywords')
        elif (inner_control_points is None) and (len(coords_by_name) == 0):
            raise ValueError('must provide either inner_control_points or individual control point keywords')

        if inner_control_points is None:
            # If points_by_name is used, ensure point names follow correct format
            self._validate_coord_names(coords_by_name.keys())

            # Build params dict from keyword arguments
            params_dict = {name: coord for name, coord in coords_by_name.items()}
        else:
            # Build params dict from inner_control_points list
            params_dict = {}
            for i, (x, y) in enumerate(inner_control_points, 1):
                params_dict[f'x{i}'] = x
                params_dict[f'y{i}'] = y

        self._dynamic_param_names = tuple(sorted(params_dict.keys(), key=lambda name: int(name[1:])))
        super().__init__(H=H, Z=Z, **params_dict)

        # degree = (num control points) - 1
        #        = (num inner points) + (2 end points) - 1
        #        = (num params / 2) + (2 end points) - 1
        #        = (num params / 2) + 1
        self.degree = len(params_dict) // 2 + 1

        self.first_control_point = (0, 0)
        self.last_control_point = (1, Z / (2 * H))

        # These are set in the following function call
        self.control_points = None
        self._spline = None

        # Updates control points and computes spline
        self._update_state()

    @property
    def PARAM_NAMES(self):
        return self._dynamic_param_names

    def psi(self, x):
        return self._spline(x)

    def derivative(self, x):
        return self._spline_derivative(x)

    def bezier_x(self, t):
        return self._bezier(t, 0)

    def bezier_y(self, t):
        return self._bezier(t, 1)

    def describe_spline_approximation(self, visualize=True):
        # Define t test sample, get x test sample
        t_test = np.linspace(0, 1, 100000)

        # Get true Bezier points
        bezier_x_test = self.bezier_x(t_test)
        bezier_y_test = self.bezier_y(t_test)

        # Get cubic spline approximation
        spline_y_test = self._spline(bezier_x_test)

        # Calculate error metrics between the real Bezier curve and the cubic spline
        diff = spline_y_test - bezier_y_test
        mse = np.mean(diff ** 2)
        avg_diff = np.mean(np.abs(diff))
        max_diff = np.max(np.abs(diff))
        x_argmax_diff = bezier_x_test[np.argmax(np.abs(diff))]
        print(f'MSE         = {mse}')
        print(f'avg y diff  = {avg_diff}')
        print(f'max y diff  = {max_diff} [x = {x_argmax_diff}]')

        if visualize:
            # Plot the real Bezier curve and the cubic spline approximation
            plt.figure(figsize=(8, 6))
            plt.plot(bezier_x_test, bezier_y_test, 'black', label='Bezier Curve', linewidth=6)
            plt.plot(bezier_x_test, spline_y_test, 'red', label='Cubic Spline Approximation', linewidth=2)
            plt.legend()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid(True)
            plt.show()

    def set_param(self, name, value):
        super().set_param(name, value)
        self._update_state()

    def set_params(self, **new_params):
        super().set_params(**new_params)
        self._update_state()

    def _update_state(self):
        self._update_control_points()
        self._compute_spline()

    def _update_control_points(self):
        inner_control_points = self._get_inner_control_points_from_params()
        self.control_points = np.array([
            self.first_control_point,
            *inner_control_points,
            self.last_control_point
        ])

    def _compute_spline(self):
        # Use higher point density in first 1% of curve since the oud has steepest slope there (2000 points total)
        t_sample = np.concatenate([
            np.linspace(0, 0.01, 1000),
            np.linspace(0.01, 1, 1000)[1:]
        ])
        x_sample = self.bezier_x(t_sample)
        y_sample = self.bezier_y(t_sample)

        self._spline = CubicSpline(x_sample, y_sample)
        self._spline_derivative = self._spline.derivative()

    def _bezier(self, t, dim):
        if dim not in {0, 1}:
            raise ValueError('dim must be 0 or 1 (0 for x, 1 for y)')

        n = self.degree
        dim_coords = self.control_points[:, dim]

        result = sum(math.comb(n, i) * (1-t)**(n-i) * t**i * dim_coords[i] for i in range(n+1))

        # Scale the result by H to get the real scale (we do this OUTSIDE of defining control points so that control
        # point definition is normalized, irrespective of H and Z)
        return result * self.H

    def _get_inner_control_points_from_params(self):
        params = self.get_params()
        return [(params[f'x{i}'], params[f'y{i}']) for i in range(1, len(params) // 2 + 1)]

    @staticmethod
    def _validate_coord_names(coord_names):
        for name in coord_names:
            if (name[0] not in {'x', 'y'}) or (not name[1:].isnumeric()):
                raise ValueError(f'coordinate keywords must follow format "x1", "y1", "x2", "y2", etc. (got "{name}" instead)')

        x_indexes = sorted(int(name[1:]) for name in coord_names if name[0] == 'x')
        y_indexes = sorted(int(name[1:]) for name in coord_names if name[0] == 'y')

        if len(x_indexes) != len(y_indexes):
            raise ValueError(f'must have corresponding x and y coordinates with matching indexes (found {len(x_indexes)} x keywords and {len(y_indexes)} y keywords)')

        expected_indexes = list(range(1, len(x_indexes) + 1))

        if (x_indexes != expected_indexes) or (y_indexes != expected_indexes):
            raise ValueError(f'x and y keywords must start at 1 and be consecutive, i.e. "x1", "y1", "x2", "y2", and so on')
