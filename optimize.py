import itertools
import textwrap

import numpy as np
from matplotlib import pyplot as plt

from models import GeneralizedGDModel, GrowthDecayModel, StaticModel, BezierModel
from util import Stopwatch, points_from_file


class Optimizer:
    def __init__(self, model, points):
        self.model = model
        self.points = np.array(points, copy=True)

        CONFIG_GETTERS = {
            GeneralizedGDModel: self._get_config__GeneralizedGDModel,
            BezierModel: self._get_config__BezierModel,
        }

        self.get_config = CONFIG_GETTERS[type(model)]

    def run(self, density, epochs=1, *, animate=False):
        if epochs == 1:
            best_params, best_mae = self.brute_force_fit(density, animate=animate)
        else:
            best_params, best_mae = self.convergent_brute_force_fit(density, epochs, animate=animate)
        self._print_results(best_params, best_mae)
        plt.show()

    def brute_force_fit(self, density, *, animate=False, ax=None, param_space_dict=None):
        model = self.model
        points = self.points
        if param_space_dict is None:
            param_space_dict, _ = self.get_config(density)

        print(f'Optimizing model:  {model}')
        print(f'Number of points to fit: {len(points):,}')

        x = points[:, 0]
        y = points[:, 1]

        # Initialize params and MAE
        best_params = model.get_params()
        best_mae = self.mae(y, model(x))

        # Setup logging
        total_iters = np.prod([len(space) for space in param_space_dict.values()])
        checkpoint = step_size = 10  # percent completion per log message

        # Get parameter space generator
        param_space = self.generate_param_space(param_space_dict)

        # Initialize animation if enabled
        if animate:
            plt.ion()
            if ax is None:
                _fig, ax = plt.subplots()
            plot_height = max(points[:, 1]) * 1.5

            def update_animation():
                ax.cla()  # clear the current axes
                wrapped_model_repr = textwrap.fill(str(model), width=60)
                model.plot(points=points, ax=ax, show=False, title=f'{wrapped_model_repr}\n\nMAE: {best_mae:.6f} mm')
                ax.title.set_fontsize(10)
                ax.set_ylim(0, plot_height)
                plt.pause(0.001)  # pause to allow the plot to update

            update_animation()

        # Iterate over parameter space and return optimal parameters
        print(f'Total iterations: {total_iters:,}')
        stopwatch = Stopwatch()
        stopwatch.start()
        for i, params in enumerate(param_space, 1):
            model.set_params(**params)

            current_mae = self.mae(y, model(x))
            if current_mae < best_mae:
                best_mae = current_mae
                best_params = params

                # Update animation if enabled
                if animate:
                    update_animation()

            if i % 10000 == 0 or i == total_iters:
                print(f'    {i:,} iters done [{round(i / total_iters * 100)}%] [{stopwatch.lap():.3f} s]  :  MAE={best_mae}, params={best_params}')

        model.set_params(**best_params)

        if animate:
            ax.title.set_weight('bold')
            plt.ioff()

        return best_params, best_mae

    def convergent_brute_force_fit(self, density, epochs, *, animate=False):
        model = self.model
        points = self.points
        param_space_dict, param_space_schedules = self.get_config(density)

        # Just to get initial baseline MAE
        x = points[:, 0]
        y = points[:, 1]

        # Initialize
        best_params = model.get_params()
        best_mae = self.mae(y, model(x))

        total_iters = np.prod([len(space) for space in param_space_dict.values()])
        print('* ' * 21)
        print('CONVERGENT BRUTE FORCE FIT'.center(40))
        print('* ' * 21)
        print()
        print(f'Number of epochs: {epochs:,}')
        print(f'Total iterations across all epochs: {total_iters * epochs:,}')
        print()

        if animate:
            _fig, ax = plt.subplots()
        else:
            ax = None

        stopwatch = Stopwatch()
        stopwatch.start()
        for epoch in range(1, epochs + 1):
            print('*' * 20 + f' EPOCH {epoch} ' + '*' * 20)
            best_params, best_mae = self.brute_force_fit(density, param_space_dict=param_space_dict, animate=animate, ax=ax)
            print()

            if epoch == epochs:
                break

            for name, value in best_params.items():
                current_space = param_space_dict[name]
                param_space_dict[name] = param_space_schedules[name](value, current_space)

        print(f'Total runtime: {stopwatch.total_elapsed():.3f} s')

        return best_params, best_mae

    @staticmethod
    def generate_param_space(param_space_dict):
        names = param_space_dict.keys()
        spaces = param_space_dict.values()
        combinations = itertools.product(*spaces)

        for combination in combinations:
            yield dict(zip(names, combination))

    @staticmethod
    def create_schedule(lower, upper, density):
        def refine_param_space(param_value, space):
            space_width = max(space) - min(space)
            margin = 1 / (density - 1)  # if density=N, we have N points and N-1 intervals, so each interval is 1/(N-1) of the space width
            refined_space = np.linspace(
                max(lower, param_value - margin * space_width),
                min(upper, param_value + margin * space_width),
                density
            )
            return np.unique(refined_space)

        return refine_param_space

    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def _print_results(self, best_params, best_mae):
        model = self.model
        print()
        print('_' * 75)
        print('RESULTS')
        print()
        print(*(f'{name.ljust(max(len(pn) for pn in model.PARAM_NAMES))} = {value}' for name, value in best_params.items()), sep='\n')
        print()
        print(model)
        print('MAE:', best_mae, 'mm')

    # Config methods ---------------------------------------------------------------------------------------------------

    def _get_config__GeneralizedGDModel(self, density):
        # Define complete param space
        param_space_dict = {
            'alpha': np.linspace(0, 1, density),
            'beta': np.linspace(0, 1, density),
            'k': np.linspace(0, 2, density)
        }

        # Define schedules for narrowing param space
        schedules = {
            'alpha': self.create_schedule(0, 1, density=density),
            'beta': self.create_schedule(0, 1, density=density),
            'k': self.create_schedule(0, 2, density=density)
        }

        return param_space_dict, schedules

    def _get_config__BezierModel(self, density):
        unit_space = np.linspace(0, 1, density)
        num_inner_control_points = self.model.degree - 1

        # Create complete param space
        param_space_dict = {'x1': [0], 'y1': unit_space}
        x_marks = np.linspace(0, 1, num_inner_control_points)
        for i in range(2, num_inner_control_points + 1):
            x_key = f'x{i}'
            y_key = f'y{i}'
            start_x_mark = x_marks[i - 2]
            end_x_mark = x_marks[i - 1]
            param_space_dict[x_key] = np.linspace(start_x_mark, end_x_mark, density)
            param_space_dict[y_key] = unit_space

        # Create schedules for narrowing param space
        schedules = {}
        for i in range(1, num_inner_control_points + 1):
            x_key = f'x{i}'
            y_key = f'y{i}'
            schedules[x_key] = self.create_schedule(0, 1, density=density)
            schedules[y_key] = self.create_schedule(0, 1, density=density)

        return param_space_dict, schedules


def main():
    # Define target points
    # target_model = BezierModel(H=500, Z=60, x1=0.0, y1=0.6252800000000002, x2=0.5055999999999999, y2=0.33664000000000005, x3=0.6624, y3=0.41472)
    # x = np.linspace(0, 500, 1000)
    # y = target_model(x)
    # points = np.column_stack((x, y))
    points = points_from_file('SANDBOX/points.txt')

    # Scale points as desired (the neck thickness will be warped, but we can still get a good model fit since the model
    # itself is mathematically fixed at (H, Z/2)
    # points = scale(points, x_max=500, y_max=185)

    #################
    USE_BEZIER = True
    #################

    # Choose param space density (per parameter) and num epochs
    if USE_BEZIER:
        density = 20
        epochs = 4
        num_inner_control_points = 3
        model = BezierModel([(i, 1) for i in np.linspace(0, 1, num_inner_control_points)], H=500, Z=60)
    else:
        density = 40
        epochs = 5
        model = GeneralizedGDModel(H=500, Z=60, alpha=0.5, beta=1, k=1)

    # Create optimizer and fit the model to the points
    optimizer = Optimizer(model, points)
    optimizer.run(density, epochs, animate=False)


if __name__ == '__main__':
    main()

'''
Optimized models (on points from file):


y_max = 200 ------------------------------------------------------------------------------------------------------------

BezierModel(H=500, Z=60, x1=0.0, y1=0.6521600000000001, x2=0.5584, y2=0.3209600000000001, x3=0.6823999999999999, y3=0.39392)
MAE: 0.5851086024146657 mm

GeneralizedGDModel(H=500, Z=60, alpha=0.44711230149622744, beta=0.9318595905100431, k=1.0008791730520934)
MAE: 1.598603000947343 mm

y_max = 185 ------------------------------------------------------------------------------------------------------------

BezierModel(H=500, Z=60, x1=0.0, y1=0.375, x2=0.1875, y2=0.390625, x3=0.53125, y3=0.484375)
MAE: 1.3052330710672473 mm

GeneralizedGDModel(H=500, Z=60, alpha=0.4406588081762854, beta=0.9506194452333127, k=0.932957842052995)
MAE: 1.4683382513168537 mm

'''
