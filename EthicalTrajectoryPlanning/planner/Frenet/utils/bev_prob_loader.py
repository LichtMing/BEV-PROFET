"""
bev_prob_loader.py
------------------
Load BEVPredProb probability maps and integrate them as a risk factor
for trajectory planning.

BEVPredProb/{k}_{timestamp_ms}/T1.npy, T2.npy, T3.npy
  - 288×288 float32 probability map covering 144m × 144m
  - Resolution: 0.5 m/pixel
  - Centered at INTERACTION world (1003, 995)
  - T1: +1s prediction, T2: +2s, T3: +3s

Coordinate transformation:
  INTERACTION → CommonRoad:  CR_x = INTER_x - 945,  CR_y = INTER_y - 993.5
  CommonRoad (x,y) → BEV pixel (row, col):
    col_288 = 2 * (cr_x + 14.5)    = 2 * cr_x + 29
    row_288 = -2 * cr_y + 148

Scenario mapping:
  BEV folder name: {k}_{timestamp_ms}
  CR_step = timestamp_ms / 100
  segment = (CR_step - 1) // 150          (0-based within recording k)
  time_step_in_scenario = (CR_step - 1) % 150   (scenario-internal, 0-based)
  scenario_id = ID_OFFSET[k] + segment
  benchmark_id = f"USA_Intersection-1_{scenario_id}_T-1"
"""

import os
import re
import numpy as np
from typing import Optional, Dict, Tuple, List

# ID offset for each recording k  (scenario_id = ID_OFFSET[k] + segment)
# k=0: 20 segments → IDs 1-20,   k=1: 18 segments → IDs 21-38, ...
ID_OFFSET = [1, 21, 39, 59, 79, 99, 119, 139]
NUM_SEGMENTS = [20, 18, 20, 20, 20, 20, 20, 17]  # segments per recording


def scenario_id_to_recording(scenario_id: int) -> Tuple[int, int]:
    """
    Convert a scenario ID to (recording_k, segment_within_recording).

    Returns:
        (k, segment) where k is 0-7, segment is the 0-based segment index.
    """
    for k in range(len(ID_OFFSET)):
        seg = scenario_id - ID_OFFSET[k]
        if 0 <= seg < NUM_SEGMENTS[k]:
            return k, seg
    raise ValueError(f"Cannot map scenario_id={scenario_id} to any recording")


def parse_benchmark_id(benchmark_id: str) -> int:
    """Extract scenario_id from benchmark_id like 'USA_Intersection-1_3_T-1'."""
    m = re.search(r'USA_Intersection-1_(\d+)_T-1', benchmark_id)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse scenario_id from benchmark_id={benchmark_id}")


def scenario_time_to_bev_timestamp(k: int, segment: int, time_step: int) -> int:
    """
    Convert (k, segment, scenario_time_step) to BEV raw timestamp_ms.

    Parameters:
        k: recording index (0-7)
        segment: 0-based segment within recording
        time_step: scenario-internal time step (0-based)

    Returns:
        raw timestamp_ms for BEV folder lookup
    """
    cr_step = segment * 150 + 1 + time_step
    return cr_step * 100


class BEVProbLoader:
    """
    Loads and queries BEVPredProb probability maps for a given scenario.

    Usage:
        loader = BEVProbLoader(bev_prob_dir, benchmark_id)
        # At each planning step:
        loader.set_time_step(time_step)
        risk = loader.compute_traj_bev_risk(trajectory, search_length)
    """

    # BEV grid parameters (global fixed, INTERACTION coordinate system)
    BEV_CENTER_X = 1003.0   # INTERACTION world x center
    BEV_CENTER_Y = 995.0    # INTERACTION world y center
    BEV_AREA_RANGE = 144.0  # meters covered
    BEV_SIZE = 288           # pixels
    BEV_RESOLUTION = BEV_AREA_RANGE / BEV_SIZE  # 0.5 m/pixel

    # CR ↔ INTERACTION coordinate offset (for USA_Intersection-1)
    X_OFFSET = 945.0
    Y_OFFSET = 993.5

    def __init__(
        self,
        bev_prob_dir: str,
        benchmark_id: str,
        bev_weight: float = 1.0,
    ):
        """
        Initialize the BEV probability loader.

        Parameters:
            bev_prob_dir: Path to BEVPredProb root directory
            benchmark_id: CommonRoad scenario benchmark_id (e.g. 'USA_Intersection-1_3_T-1')
            bev_weight: Scaling weight for BEV risk contribution
        """
        self.bev_prob_dir = bev_prob_dir
        self.benchmark_id = benchmark_id
        self.bev_weight = bev_weight

        # Parse scenario ID and determine recording/segment
        try:
            self.scenario_id = parse_benchmark_id(benchmark_id)
            self.k, self.segment = scenario_id_to_recording(self.scenario_id)
        except ValueError:
            print(f"[BEVProbLoader] Warning: benchmark_id '{benchmark_id}' not a USA_Intersection-1 scenario. "
                  f"BEV risk disabled.")
            self.k = -1
            self.segment = -1
            self.available_timestamps = {}
            return

        # Scan available BEV folders for this recording and segment
        self.available_timestamps = self._scan_available_timestamps()

        # Cache for loaded probability maps
        self._cached_ts: Optional[int] = None
        self._cached_maps: Optional[List[np.ndarray]] = None  # [T1, T2, T3]

        print(f"[BEVProbLoader] scenario={benchmark_id}, k={self.k}, segment={self.segment}, "
              f"available BEV timestamps: {len(self.available_timestamps)}")

    def _scan_available_timestamps(self) -> Dict[int, str]:
        """
        Scan BEVPredProb directory for folders that belong to this scenario's
        time range.

        Returns:
            Dict mapping scenario_time_step → folder_path
        """
        if self.k < 0 or not os.path.isdir(self.bev_prob_dir):
            return {}

        ts_map = {}
        prefix = f"{self.k}_"

        # CR step range for this scenario
        cr_start = self.segment * 150 + 1   # first CR step of scenario
        cr_end = (self.segment + 1) * 150   # last CR step of scenario

        for folder_name in os.listdir(self.bev_prob_dir):
            if not folder_name.startswith(prefix):
                continue
            try:
                timestamp_ms = int(folder_name[len(prefix):])
            except ValueError:
                continue

            cr_step = timestamp_ms // 100
            if cr_start <= cr_step <= cr_end:
                scenario_ts = cr_step - cr_start  # 0-based scenario time step
                folder_path = os.path.join(self.bev_prob_dir, folder_name)
                if os.path.isdir(folder_path):
                    ts_map[scenario_ts] = folder_path

        return ts_map

    def _load_prob_maps(self, folder_path: str) -> List[np.ndarray]:
        """Load T1.npy, T2.npy, T3.npy from a BEVPredProb folder."""
        maps = []
        for t_file in ["T1.npy", "T2.npy", "T3.npy"]:
            path = os.path.join(folder_path, t_file)
            if os.path.exists(path):
                maps.append(np.load(path))
            else:
                maps.append(np.zeros((self.BEV_SIZE, self.BEV_SIZE), dtype=np.float32))
        return maps

    def _find_nearest_timestamp(self, time_step: int) -> Optional[int]:
        """
        Find the nearest available BEV timestamp for the given scenario time_step.

        Returns:
            The nearest available scenario_time_step, or None if no data available.
        """
        if not self.available_timestamps:
            return None

        available = sorted(self.available_timestamps.keys())
        # Find closest
        best = min(available, key=lambda t: abs(t - time_step))
        # Only use if within 5 seconds (50 steps)
        if abs(best - time_step) <= 50:
            return best
        return None

    def set_time_step(self, time_step: int) -> bool:
        """
        Set current scenario time step and load the nearest BEV probability maps.

        Returns:
            True if BEV maps are available for this time step.
        """
        nearest = self._find_nearest_timestamp(time_step)
        if nearest is None:
            self._cached_ts = None
            self._cached_maps = None
            return False

        if self._cached_ts != nearest:
            folder_path = self.available_timestamps[nearest]
            self._cached_maps = self._load_prob_maps(folder_path)
            self._cached_ts = nearest

        return True

    def cr_to_bev_pixel(self, cr_x: float, cr_y: float) -> Tuple[int, int]:
        """
        Convert CommonRoad world coordinates to BEV 288×288 pixel (row, col).

        Parameters:
            cr_x, cr_y: CommonRoad world coordinates

        Returns:
            (row, col) in the 288×288 BEV grid
        """
        # CR → INTERACTION
        inter_x = cr_x + self.X_OFFSET
        inter_y = cr_y + self.Y_OFFSET

        # INTERACTION → BEV pixel (144×144 grid, then 2x upsample)
        col_144 = (inter_x - self.BEV_CENTER_X) + self.BEV_AREA_RANGE / 2 + 0.5
        row_144 = -(inter_y - self.BEV_CENTER_Y) + self.BEV_AREA_RANGE / 2 + 0.5

        col_288 = int(col_144 * 2)
        row_288 = int(row_144 * 2)

        return row_288, col_288

    def query_prob(self, cr_x: float, cr_y: float, t_index: int) -> float:
        """
        Query BEV probability at a CommonRoad world position for a given
        prediction time index.

        Parameters:
            cr_x, cr_y: CommonRoad world coordinates
            t_index: prediction time index (0=T1, 1=T2, 2=T3)

        Returns:
            Probability value in [0, 1], or 0 if out of bounds or no data.
        """
        if self._cached_maps is None or t_index < 0 or t_index >= 3:
            return 0.0

        row, col = self.cr_to_bev_pixel(cr_x, cr_y)

        if 0 <= row < self.BEV_SIZE and 0 <= col < self.BEV_SIZE:
            return float(self._cached_maps[t_index][row, col])
        return 0.0

    def compute_traj_bev_risk(
        self,
        traj: list,
        search_length: int,
        dt: float = 0.1,
    ) -> float:
        """
        Compute BEV-based risk for a full trajectory (spanning up to 3
        InteractionMap periods).

        For each trajectory point, determine which T prediction (T1/T2/T3)
        is most appropriate based on the point's future time, look up the
        BEV probability, and sum them.

        Parameters:
            traj: List of trajectory points, each as [x, y, s, d, v, yaw].
                  Length ≤ 3 * search_length.
            search_length: Number of points per InteractionMap period.
            dt: Time step between trajectory points (default 0.1s).

        Returns:
            Weighted sum of BEV probabilities along the trajectory.
        """
        if self._cached_maps is None:
            return 0.0

        risk_sum = 0.0
        for i, point in enumerate(traj):
            x, y = point[0], point[1]

            # Determine future time in seconds from current ego time
            future_time_s = (i + 1) * dt

            # Map future time to T prediction index:
            #   T1 = +1s (best for 0-1.5s)
            #   T2 = +2s (best for 1.5-2.5s)
            #   T3 = +3s (best for 2.5-3.5s)
            if future_time_s <= 1.5:
                t_idx = 0   # T1
            elif future_time_s <= 2.5:
                t_idx = 1   # T2
            elif future_time_s <= 3.5:
                t_idx = 2   # T3
            else:
                # Beyond T3 prediction range, use T3 with decay
                t_idx = 2

            prob = self.query_prob(x, y, t_idx)
            risk_sum += prob * self.bev_weight

        return risk_sum

    def compute_segment_bev_risk(
        self,
        traj_segment: list,
        t_index: int,
    ) -> float:
        """
        Compute BEV risk for a single InteractionMap segment using a fixed
        T prediction index.

        Parameters:
            traj_segment: List of trajectory points [x, y, s, d, v, yaw].
            t_index: Which T prediction to use (0=T1, 1=T2, 2=T3).

        Returns:
            Sum of BEV probabilities along the segment.
        """
        if self._cached_maps is None:
            return 0.0

        risk_sum = 0.0
        for point in traj_segment:
            x, y = point[0], point[1]
            prob = self.query_prob(x, y, t_index)
            risk_sum += prob * self.bev_weight

        return risk_sum

    @property
    def is_available(self) -> bool:
        """Check if BEV probability data is available for this scenario."""
        return self.k >= 0 and len(self.available_timestamps) > 0
