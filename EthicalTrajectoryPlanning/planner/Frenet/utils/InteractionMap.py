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
    """Risk accumulation map with optional adaptive (non-uniform) resolution.

    When ``adaptive_resolution=True``, the ``resolution_mode`` selects the
    coordinate compression strategy:

    **'bands'** (default) — Piecewise-linear bands.
        +-----------+----------+------------------+
        | Distance  | m / pixel| Coverage         |
        +-----------+----------+------------------+
        |  0 – 20 m |  0.5     | ego vicinity     |
        | 20 – 50 m |  1.5     | mid-range        |
        | 50 –100 m |  4.0     | far periphery    |
        +-----------+----------+------------------+

    **'linear'** — Grid cell size grows linearly with distance:
        Δs(d) = base + growth_rate · d.  The resulting (x,y)→(row,col)
        mapping is logarithmic compression:  px(d) = (1/b)·ln(1+b·d/a).

    **'speed'** — Bands + speed-coupled risk inflation.
        Band radii scale with ego velocity: r_i *= (1 + scale · v_ego).
        In ``update_map``, risk spreading width and risk value are
        amplified by the tracked obstacle's instantaneous speed.

    When ``adaptive_resolution=False`` the map behaves identically to
    the original uniform grid (``resolution_mode`` is ignored).
    """

    def __init__(
            self,
            bev_map: np.ndarray,
            anchor_state,
            size: float,
            obs_width: float,
            update_length: int,
            adaptive_resolution: bool = True,
            res_bands: list = None,
            resolution_mode: str = 'bands',
            linear_growth_rate: float = 0.03,
            ego_velocity: float = 0.0,
            speed_band_scale: float = 0.02,
            speed_risk_factor: float = 0.02,
            max_distance: float = None,
    ):
        self.anchor_state = anchor_state
        self.size = size                     # finest resolution (m/px)
        self.update_length = update_length
        self.obs_width = obs_width
        self.adaptive = adaptive_resolution
        self.ego_center = anchor_state.position.copy()
        self.resolution_mode = resolution_mode
        self.linear_growth_rate = linear_growth_rate
        self.ego_velocity = ego_velocity
        self.speed_band_scale = speed_band_scale
        self.speed_risk_factor = speed_risk_factor

        # ── Resolution bands ──
        # Each entry: (radius_m, meters_per_pixel)
        # Must be sorted by radius ascending. The last band extends to ∞.
        if res_bands is None:
            res_bands = [
                (20.0, size),          # 0.5 m/px
                (50.0, size * 3),      # 1.5 m/px
                (100.0, size * 8),     # 4.0 m/px
            ]
        self.res_bands = list(res_bands)  # copy to avoid mutation

        # max_distance: coverage radius (default = last band radius)
        if max_distance is not None:
            self.max_distance = max_distance
        else:
            self.max_distance = self.res_bands[-1][0]

        if self.adaptive:
            if resolution_mode == 'linear':
                self._build_linear_mapping()
            elif resolution_mode == 'speed':
                self._build_speed_mapping()
            else:  # 'bands'
                self._build_nonlinear_mapping()
            grid_side = self._half_pixels * 2
            self.risk_map = np.zeros((grid_side, grid_side), dtype=np.float64)
            self.tree_map = np.zeros((grid_side, grid_side), dtype=np.float64)
            self.visited_map = np.full((grid_side, grid_side), 0.001, dtype=np.float64)
            self.risk_map_vertical = np.zeros((grid_side, grid_side), dtype=np.float64)
            self.visited_map_vertical = np.full((grid_side, grid_side), 0.001, dtype=np.float64)
            self.bev_map = np.zeros((grid_side, grid_side), dtype=bev_map.dtype)
            # length/width in metres that the grid covers
            self.length = self.max_distance * 2
            self.width = self.length
        else:
            # Original uniform behaviour
            self.bev_map = bev_map
            self.risk_map = np.full_like(bev_map, fill_value=0, dtype=np.float64)
            self.tree_map = np.full_like(bev_map, fill_value=0, dtype=np.float64)
            self.visited_map = np.full_like(bev_map, fill_value=0.001, dtype=np.float64)
            self.risk_map_vertical = np.full_like(bev_map, fill_value=0, dtype=np.float64)
            self.visited_map_vertical = np.full_like(bev_map, fill_value=0.001, dtype=np.float64)
            self.length, self.width = np.array(bev_map.shape) * size

        self.origin = anchor_state.position - np.array([self.length, self.width]) / 2
        self.rotation = anchor_state.orientation
        self.res = self.length / self.size

        colors = ["#739D2E", "#87BB33", "#FEE599", "#EA700D", "#FF3C3C"]
        nodes = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

    # ── Non-linear coordinate mapping ──────────────────────────────────

    def _build_nonlinear_mapping(self):
        """Pre-compute cumulative pixel offsets at each band boundary."""
        self._band_cumul_px = []
        cumul = 0.0
        prev_r = 0.0
        for radius_m, mpp in self.res_bands:
            cumul += (radius_m - prev_r) / mpp
            self._band_cumul_px.append(cumul)
            prev_r = radius_m
        self._half_pixels = int(math.ceil(cumul)) + 1

    def _build_linear_mapping(self):
        """Build mapping for linear-growth mode: Δs(d) = a + b·d.

        The forward transform is logarithmic compression:
            pixel(d) = (1/b)·ln(1 + b·d/a)
        Inverse:
            d(pixel) = (a/b)·(exp(b·pixel) − 1)
        """
        a = self.size
        b = self.linear_growth_rate
        max_d = self.max_distance
        if b < 1e-12:
            # Degenerate: growth ≈ 0 → uniform grid
            self._half_pixels = int(math.ceil(max_d / a)) + 1
        else:
            self._half_pixels = int(math.ceil(
                (1.0 / b) * math.log(1.0 + b * max_d / a))) + 1

    def _build_speed_mapping(self):
        """Scale band radii by ego velocity, then build bands mapping.

        scale = 1 + speed_band_scale × v_ego
        Each band radius r_i → r_i × scale (resolution m/px unchanged).
        This expands coverage at higher speeds while keeping pixel count
        proportional.
        """
        scale = 1.0 + self.speed_band_scale * self.ego_velocity
        self.res_bands = [(r * scale, mpp) for r, mpp in self.res_bands]
        self.max_distance = self.res_bands[-1][0]
        self._build_nonlinear_mapping()

    # ── Dispatch helpers ──────────────────────────────────────────────

    def _coord_to_pixel_1d(self, d: float) -> int:
        """Map signed distance (m) from ego → pixel index."""
        if self.resolution_mode == 'linear':
            return self._coord_to_pixel_linear_1d(d)
        return self._coord_to_pixel_bands_1d(d)

    def _coord_to_pixel_bands_1d(self, d: float) -> int:
        """Piecewise-linear band compression."""
        sign = 1 if d >= 0 else -1
        ad = abs(d)
        prev_r = 0.0
        cumul = 0.0
        for i, (radius_m, mpp) in enumerate(self.res_bands):
            if ad <= radius_m:
                cumul += (ad - prev_r) / mpp
                return int(sign * cumul + self._half_pixels)
            cumul += (radius_m - prev_r) / mpp
            prev_r = radius_m
        # Beyond last band — extrapolate with last resolution
        cumul += (ad - prev_r) / self.res_bands[-1][1]
        return int(sign * cumul + self._half_pixels)

    def _coord_to_pixel_linear_1d(self, d: float) -> int:
        """Logarithmic compression: Δs(d) = a + b·d."""
        sign = 1 if d >= 0 else -1
        ad = abs(d)
        a = self.size
        b = self.linear_growth_rate
        if b < 1e-12:
            px = ad / a
        else:
            px = (1.0 / b) * math.log(1.0 + b * ad / a)
        return int(sign * px + self._half_pixels)

    def _pixel_to_coord_1d(self, px: int) -> float:
        """Inverse: pixel index → signed distance in m."""
        if self.resolution_mode == 'linear':
            return self._pixel_to_coord_linear_1d(px)
        return self._pixel_to_coord_bands_1d(px)

    def _pixel_to_coord_bands_1d(self, px: int) -> float:
        """Inverse for piecewise-linear bands."""
        p = px - self._half_pixels
        sign = 1 if p >= 0 else -1
        ap = abs(p)
        prev_r = 0.0
        cumul = 0.0
        for radius_m, mpp in self.res_bands:
            band_px = (radius_m - prev_r) / mpp
            if ap <= cumul + band_px:
                return sign * (prev_r + (ap - cumul) * mpp)
            cumul += band_px
            prev_r = radius_m
        return sign * (prev_r + (ap - cumul) * self.res_bands[-1][1])

    def _pixel_to_coord_linear_1d(self, px: int) -> float:
        """Inverse for linear-growth mode: d = (a/b)·(exp(b·p) − 1)."""
        p = px - self._half_pixels
        sign = 1 if p >= 0 else -1
        ap = abs(p)
        a = self.size
        b = self.linear_growth_rate
        if b < 1e-12:
            return sign * ap * a
        return sign * (a / b) * (math.exp(b * ap) - 1.0)

    def _mpp_at_pixel_1d(self, px: int) -> float:
        """Return the metres-per-pixel at a given pixel index."""
        if self.resolution_mode == 'linear':
            return self._mpp_at_pixel_linear_1d(px)
        return self._mpp_at_pixel_bands_1d(px)

    def _mpp_at_pixel_bands_1d(self, px: int) -> float:
        """Metres-per-pixel for piecewise-linear bands."""
        p = abs(px - self._half_pixels)
        cumul = 0.0
        prev_r = 0.0
        for radius_m, mpp in self.res_bands:
            band_px = (radius_m - prev_r) / mpp
            if p <= cumul + band_px:
                return mpp
            cumul += band_px
            prev_r = radius_m
        return self.res_bands[-1][1]

    def _mpp_at_pixel_linear_1d(self, px: int) -> float:
        """Metres-per-pixel for linear-growth mode: mpp = a + b·|d|."""
        d = abs(self._pixel_to_coord_linear_1d(px))
        return self.size + self.linear_growth_rate * d

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
        grid_max = self.risk_map.shape[0]
        risk_after_collision = []
        is_speed_mode = (self.resolution_mode == 'speed')
        for i in range(update_range, 0, -1):
            p1 = trajectory.get_global_state(i).position
            p2 = trajectory.get_global_state(i-1).position
            v1 = trajectory.get_global_state(i).velocity

            # Speed-coupled risk amplification
            if is_speed_mode:
                speed_risk_mult = 1.0 + self.speed_risk_factor * v1
                speed_w_bonus = max(0.0, v1 / 5.0)  # +1 px width per 5 m/s
            else:
                speed_risk_mult = 1.0
                speed_w_bonus = 0.0
            effective_risk = grid_risk * speed_risk_mult

            if self.in_map_check(p1) and self.in_map_check(p2):

                pixel_position_1 = self.position_to_pixel(p1)
                pixel_position_2 = self.position_to_pixel(p2)

                if pixel_position_1 == pixel_position_2:
                    r, c = pixel_position_1
                    if 0 <= r < grid_max and 0 <= c < grid_max:
                        self.risk_map[r, c] += effective_risk
                        self.visited_map[r, c] += 1
                else:
                    if trajectory.uncertainty_list is not None and grid_risk > 0.1:
                        unc_idx = min(i - 1, len(trajectory.uncertainty_list) - 1)
                        rr, cc, vals = weighted_line(*pixel_position_1, *pixel_position_2, w=4+self.confidence_width(trajectory.uncertainty_list[unc_idx]) + speed_w_bonus,
                                                     rmin=0, rmax=grid_max)
                    else:
                        rr, cc, vals = weighted_line(*pixel_position_1, *pixel_position_2, w=5 + speed_w_bonus, rmin=0, rmax=grid_max)

                    # Clamp columns to valid range
                    mask = (cc >= 0) & (cc < grid_max)
                    rr, cc, vals = rr[mask], cc[mask], vals[mask]

                    self.risk_map[rr, cc] += effective_risk * vals
                    self.visited_map[rr, cc] += 1

                if draw_tree_ax is not None:
                    draw_tree_ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c=self.cmap(grid_risk - 0.001 if grid_risk == 1 else grid_risk), linewidth=2.5, zorder=25)
                if trajectory.collision_step != -1 and i > trajectory.collision_step:
                    risk_after_collision.append(effective_risk)
                    grid_risk = grid_risk
                else:
                    risk_after_collision.append(effective_risk)
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
        if self.adaptive:
            d = position - self.ego_center
            return [self._coord_to_pixel_1d(d[0]),
                    self._coord_to_pixel_1d(d[1])]
        relative_position = position - self.origin
        pixel_position = [int(coordinate // self.size) for coordinate in relative_position]
        return pixel_position

    def rel_position_to_pixel(self, rel_position):
        if self.adaptive:
            return [self._coord_to_pixel_1d(rel_position[0]),
                    self._coord_to_pixel_1d(rel_position[1])]
        pixel_position = [int(coordinate // self.size + self.res / 2) for coordinate in rel_position]
        return pixel_position

    def in_map_check(self, position):
        if self.adaptive:
            d = position - self.ego_center
            return abs(d[0]) < self.max_distance and abs(d[1]) < self.max_distance
        diff = position - self.origin
        if max(diff) >= self.length or min(diff) < 0:
            return False
        else:
            return True

    def in_map_check_rel(self, rel_position):
        if self.adaptive:
            return abs(rel_position[0]) < self.max_distance and abs(rel_position[1]) < self.max_distance
        if max(abs(rel_position)) > self.length / 2:
            return False
        else:
            return True

    def pixel_to_position(self, pixel):
        if self.adaptive:
            return self.ego_center + np.array([
                self._pixel_to_coord_1d(pixel[0]),
                self._pixel_to_coord_1d(pixel[1])])
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
        if self.adaptive:
            # Non-linear pixel → world position + variable marker sizes
            rows, cols = pixel_to_plot
            gx = np.array([self._pixel_to_coord_1d(int(r)) for r in rows]) + self.ego_center[0]
            gy = np.array([self._pixel_to_coord_1d(int(c)) for c in cols]) + self.ego_center[1]
            global_positions = (gx, gy)
            # Each pixel's marker area scales with the square of its local
            # meters-per-pixel so that cells visually fill their physical area.
            base_s = (200. / fig.dpi) ** 2   # base size for 0.5 m/px
            mpp_row = np.array([self._mpp_at_pixel_1d(int(r)) for r in rows])
            mpp_col = np.array([self._mpp_at_pixel_1d(int(c)) for c in cols])
            mpp_max = np.maximum(mpp_row, mpp_col)
            sizes = base_s * (mpp_max / self.size) ** 2
        else:
            global_positions = pixel_to_plot * np.array(self.size) + self.origin[:, np.newaxis]
            global_positions = (global_positions[0], global_positions[1])
            sizes = (200. / fig.dpi) ** 2
        a1 = ax.scatter(global_positions[0], global_positions[1], s=sizes, marker='s', c=self.cmap(self.risk_map[pixel_to_plot]), alpha=0.8, zorder=25)
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
        grid_max = self.risk_map.shape[0]
        for (x, y, s, d, v, yaw) in traj:
            pos = np.asarray([x, y])
            if self.in_map_check(pos):
                pixel_pos = self.position_to_pixel(pos)
                r, c = pixel_pos[0], pixel_pos[1]
                if 0 <= r < grid_max and 0 <= c < grid_max:
                    risk_sum += self.risk_map[r, c]
        return risk_sum