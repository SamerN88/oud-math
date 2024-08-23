import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.widgets import Button
from models import GeneralizedGDModel, BezierModel, GrowthDecayModel
from procedure import RibProcedure, OudSpecs
from optimize import Optimizer

# TODO: review this file


class PointSelector:
    def __init__(self, image_path, *, H, Z):
        self.image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.H = H
        self.Z = Z

        self.fig, self.ax = None, None

        self._segments = []
        self._points = []

        self._is_drawing = False
        self._stroke_width = 3  # set the width for both lines and points
        self._start_point = None
        self._current_segment = []
        self._cid_click = None
        self._cid_release = None
        self._cid_motion = None

    def run(self):
        self.init_plot()

        # Create the Undo button
        undo_ax = plt.axes((0.05, 0.05, 0.1, 0.075))
        undo_button = Button(undo_ax, 'Undo')
        undo_button.on_clicked(self.undo)

        # Create the Done button
        done_ax = plt.axes((0.81, 0.05, 0.1, 0.075))
        done_button = Button(done_ax, 'Done')
        done_button.on_clicked(self.done)

        self._cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self._cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self._cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        plt.show()
        self.fig.canvas.mpl_disconnect(self._cid_click)
        self.fig.canvas.mpl_disconnect(self._cid_release)
        self.fig.canvas.mpl_disconnect(self._cid_motion)

        return self.process_points()

    def on_press(self, event):
        if event.inaxes == self.ax:  # ensure the click is within the plot axes
            self._is_drawing = True
            self._start_point = (int(event.xdata), int(event.ydata))
            self._current_segment = [self._start_point]
            self.update_plot()

    def on_release(self, event):
        if event.inaxes == self.ax:  # ensure the release is within the plot axes
            self._is_drawing = False
            self._segments.append(self._current_segment.copy())
            self._current_segment = []

    def on_motion(self, event):
        if self._is_drawing and event.inaxes == self.ax:
            if event.xdata is not None and event.ydata is not None:
                x, y = int(event.xdata), int(event.ydata)
                self._current_segment.append((x, y))
                self.update_plot()

    def init_plot(self):
        # Calculate aspect ratio of the image
        aspect_ratio = self.image.shape[1] / self.image.shape[0]
        width = 12  # choose some base width
        height = (width / aspect_ratio) + 1  # the +1 is to add space for the buttons

        # Instantiate the plot
        self.fig, self.ax = plt.subplots(figsize=(width, height))

        self.style_plot()

    def style_plot(self):
        self.ax.set_title('NOTE: Make sure to select the endpoints.')
        self.ax.title.set_weight('bold')
        self.ax.set_xticks([])  # remove x-axis ticks
        self.ax.set_yticks([])  # remove y-axis ticks
        self.ax.imshow(self.image)  # show image

    def update_plot(self):
        self.ax.cla()  # clear the axes
        self.style_plot()

        for segment in self._segments + [self._current_segment]:
            if len(segment) == 0:
                continue
            elif len(segment) == 1:
                self.ax.scatter(*zip(*segment), c='red', s=self._stroke_width)
            else:
                self.ax.plot(*zip(*segment), c='red', linewidth=self._stroke_width)

        plt.draw()

    def undo(self, event):
        if self._segments:
            self._segments.pop()
            self.update_plot()

    def done(self, event):
        for segment in self._segments:
            self._points.extend(segment)
        plt.close(self.fig)

    def process_points(self):
        points = np.array(self._points, dtype=float)

        # Adjust coordinates to standard Cartesian plane, with oud base at (0, 0)
        points[:, 0] -= min(points[:, 0])
        points[:, 1] *= -1
        points[:, 1] -= min(points[:, 1])

        # Scale points according to H, and transform so that depth is preserved and endpoint is at (H, Z/2)

        points = points[points[:, 0].argsort()]  # sort
        points *= self.H / max(points[:, 0])  # scale

        depth = max(points[:, 1])  # get original depth
        neck_thickness = self.Z / 2  # get desired neck thickness

        # In each iteration, we translate points vertically on a linear scale so the rightmost point is at (H, Z/2),
        # then we scale all points vertically so that the highest point is at the original oud depth. However, doing
        # either of these transformations will partially distort the previous transformation; but by doing both
        # transformations repeatedly, both the rightmost point and the highest point will converge to the desired
        # values. In other words, with each iteration, the extent to which one transformation distorts the other
        # diminishes, so we can keep applying the transformations until the extent of the distortion is practically
        # zero.
        for _ in range(50):
            diff = neck_thickness - points[-1, 1]
            points[:, 1] += np.array([(diff / self.H) * x for x in points[:, 0]])
            points[:, 1] *= depth / max(points[:, 1])

        return points


def main():
    # Usage

    model = GeneralizedGDModel(H=500, Z=60, alpha=0.5, beta=1, k=1)
    density = 50

    # model = BezierModel([(x, 0.5) for x in np.linspace(0, 1, 3)], H=500, Z=60)
    # density = 6

    selector = PointSelector('assets/img/oud_profile.png', H=model.H, Z=model.Z)
    points = selector.run()

    optimizer = Optimizer(model, points)
    optimizer.run(density, epochs=3, animate=True)

    specs = OudSpecs(model)
    specs.make_specs_pdf('oud_specs.pdf')


if __name__ == '__main__':
    main()
