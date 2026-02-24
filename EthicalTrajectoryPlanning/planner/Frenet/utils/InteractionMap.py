import math
import os.path
import sys
import cv2
import matplotlib.pyplot
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from planner.Frenet.utils.frenet_functions import FrenetTrajectory
from skimage.draw import line, line_aa
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from planner.Frenet.utils.helper_functions import green_to_red_colormap, abs_to_rel_coord
import alphashape

def trapez(y,y0,w):
    return np.clip(np.minimum(y+1+w/2-y0, -y+1+w/2+y0),0,1)

def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1-c0) < abs(r1-r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)

    # The following is now always < 1 in abs
    try:
        slope = (r1-r0) / (c1-c0)
    except Exception as e:
        print("error")

    # Adjust weight by the slope
    w *= np.sqrt(1+np.abs(slope)) / 2

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1+1, dtype=float)
    y = x * slope + (c1*r0-c0*r1) / (c1-c0)

    # Now instead of 2 values for y, we have 2*np.ceil(w/2).
    # All values are 1 except the upmost and bottommost.
    thickness = np.ceil(w/2)
    yy = (np.floor(y).reshape(-1,1) + np.arange(-thickness-1,thickness+2).reshape(1,-1))
    xx = np.repeat(x, yy.shape[1])
    vals = trapez(yy, y.reshape(-1,1), w).flatten()

    yy = yy.flatten()

    # Exclude useless parts and those outside of the interval
    # to avoid parts outside of the picture
    mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

    return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])

class InteractionMap(object):
    def __init__(
            self,
            bev_map: np.ndarray,
            anchor_state,
            size: float,
            obs_width: float,
            update_length: int,
    ):
        self.bev_map = bev_map
        self.risk_map = np.full_like(bev_map, fill_value=0, dtype=np.float64)
        self.tree_map = np.full_like(bev_map, fill_value=0, dtype=np.float64)
        self.visited_map = np.full_like(bev_map, fill_value=0.001, dtype=np.float64)
        self.origin = anchor_state.position - np.array(bev_map.shape) * size / 2
        self.length, self.width = np.array(bev_map.shape) * size
        self.size = size
        self.update_length = update_length
        self.anchor_state = anchor_state
        colors = ["#739D2E", "#87BB33", "#FEE599", "#EA700D", "#FF3C3C"]
        nodes = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))
        # self.cmap = plt.get_cmap('RdYlGn_r')
        # align with the walenet
        self.risk_map_vertical = np.full_like(bev_map, fill_value=0, dtype=np.float64)
        self.visited_map_vertical = np.full_like(bev_map, fill_value=0.001, dtype=np.float64)
        self.rotation = anchor_state.orientation
        self.res = self.length / self.size
        self.obs_width = obs_width

    def update_map(
            self,
            trajectory: FrenetTrajectory,
            risk_value: float,
            update_range: float = None,
            draw_tree_ax: matplotlib.pyplot.Axes = None,
    ):
        # if the collision position is out of map, the vehicle cannot detect this risk

        if trajectory.collision_step != -1:
            if not self.in_map_check(trajectory.get_global_state(trajectory.collision_step).position):
                risk_value = 0
        if update_range is None:
            update_range = self.update_length
        grid_risk = risk_value
        risk_after_collision = []
        for i in range(update_range, 0, -1):
            p1 = trajectory.get_global_state(i).position
            p2 = trajectory.get_global_state(i-1).position
            v1 = trajectory.get_global_state(i).velocity
            if self.in_map_check(p1) and self.in_map_check(p2):

                pixel_position_1 = self.position_to_pixel(p1)
                pixel_position_2 = self.position_to_pixel(p2)

                if pixel_position_1 == pixel_position_2:
                    self.risk_map[pixel_position_1[0], pixel_position_1[1]] += grid_risk
                    self.visited_map[pixel_position_1[0], pixel_position_1[1]] += 1
                else:
                    if trajectory.uncertainty_list is not None and grid_risk > 0.1:
                        unc_idx = min(i - 1, len(trajectory.uncertainty_list) - 1)
                        rr, cc, vals = weighted_line(*pixel_position_1, *pixel_position_2, w=4+self.confidence_width(trajectory.uncertainty_list[unc_idx]),
                                                     rmin=0, rmax=400)
                    else:
                        rr, cc, vals = weighted_line(*pixel_position_1, *pixel_position_2, w=5, rmin=0, rmax=400)

                    # rr, cc, vals = line_aa(*pixel_position_1, *pixel_position_2)
                    # rr = rr[1:]
                    # cc = cc[1:]
                    self.risk_map[rr, cc] += grid_risk * vals
                    self.visited_map[rr, cc] += 1

                if draw_tree_ax is not None:
                    draw_tree_ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c=self.cmap(grid_risk - 0.001 if grid_risk == 1 else grid_risk), linewidth=2.5, zorder=25)
                if trajectory.collision_step != -1 and i > trajectory.collision_step:
                    risk_after_collision.append(grid_risk)
                    grid_risk = grid_risk
                else:
                    risk_after_collision.append(grid_risk)
                    grid_risk = grid_risk * max(0.8 - v1 / 20, 0.2)
        return sum(risk_after_collision) / update_range if update_range > 0 else 0

    def confidence_width(self, cov):
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*.

        Parameters
        ----------
        x, y : array-like, shape (n, )
            Input data.

        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.

        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.

        **kwargs
            Forwarded to `~matplotlib.patches.Ellipse`

        Returns
        -------
        matplotlib.patches.Ellipse
        """

        pearson = cov[0][1] / (np.sqrt(cov[0][0] * cov[1][1]) + sys.float_info.epsilon)
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)

        return max(ell_radius_x, ell_radius_y)

    def position_to_pixel(self, position):
        relative_position = position - self.origin
        pixel_position = [int(coordinate // self.size) for coordinate in relative_position]
        return pixel_position

    def rel_position_to_pixel(self, rel_position):
        pixel_position = [int(coordinate // self.size + self.res / 2) for coordinate in rel_position]
        return pixel_position

    def in_map_check(self, position):
        diff = position - self.origin
        if max(diff) >= self.length or min(diff) < 0:
            return False
        else:
            return True

    def in_map_check_rel(self, rel_position):
        if max(abs(rel_position)) > self.length / 2:
            return False
        else:
            return True

    def pixel_to_position(self, pixel):
        return self.origin + pixel * self.size

    def get_map(self):
        return self.risk_map / self.visited_map

    def save_map_vertical(self, file):
        img = self.risk_map_vertical / self.visited_map_vertical * 128
        img[np.where(self.visited_map_vertical> 1)] += 127
        cv2.imwrite(filename=file, img=img.astype(np.int16))

    def map_post_processing(self):
        chull = convex_hull_image(self.visited_map > 1)
        not_visited = self.visited_map < 1
        pixels = np.where(not_visited & chull)
        self.visited_map[pixels] += 1

    def draw_map(
            self,
            fig,
            ax: matplotlib.pyplot.Axes,
            filename,
            best_traj,
            worst_traj,
            ):
        # self.map_post_processing()
        pixel_to_plot = np.where(self.visited_map > 0.001)
        plot_map = np.where(self.visited_map > 0.001, 1, 0)
        global_positions = pixel_to_plot * np.array(self.size) + self.origin[:, np.newaxis]
        a1 = ax.scatter(global_positions[0], global_positions[1], s=(200./fig.dpi)**2, marker='s', c=self.cmap(self.risk_map[pixel_to_plot]), alpha=0.8, zorder=25)
        # try:
        #     shape = alphashape.alphashape(list(zip(global_positions[0], global_positions[1])))
        # except:
        #     print("hello")
        # shape_x, shape_y = shape.exterior.coords.xy
        # ax.plot(shape_x, shape_y, 'o', color='red', markersize=4)
        # plt.colorbar(a1, ax=ax)
        # self.draw_traj(best_traj, worst_traj, ax)
        filepath = os.path.join("../../saved_fig", filename)
        dir_path = os.path.split(filepath)[0]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        fig.savefig(filepath, dpi=900, pad_inches=0.0)
        plt.close(fig)
        # print("hello")

    def draw_traj(self, best_traj: [[float]], worst_traj: [[float]],
                  ax: matplotlib.pyplot.Axes,):
        pos_x = [x for (x, y, s, d, v, yaw) in best_traj]
        pos_y = [y for (x, y, s, d, v, yaw) in best_traj]
        ax.plot(pos_x, pos_y, c='#4682e6', linewidth=3.0)
        # pos_x = [x for (x, y, s, d, v) in worst_traj]
        # pos_y = [y for (x, y, s, d, v) in worst_traj]
        # ax.plot(pos_x, pos_y, c='red', linewidth=1)

    def sum_traj_risk(self, traj: [[float]]):
        risk_sum = 0
        for (x, y, s, d, v, yaw) in traj:
            pos = np.asarray([x, y])
            if self.in_map_check(pos):
                pixel_pos = self.position_to_pixel(pos)
                risk_sum += self.risk_map[pixel_pos[0], pixel_pos[1]]
        return risk_sum