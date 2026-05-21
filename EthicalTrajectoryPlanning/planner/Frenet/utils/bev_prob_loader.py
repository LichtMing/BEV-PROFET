"""
bev_prob_loader.py
------------------
Load BEVPredProb probability maps and integrate them as a risk factor
for trajectory planning.

Supported scenarios:
  - USA_Intersection-1: BEVPredProb/{k}_{timestamp_ms}/T1.npy, T2.npy, T3.npy
  - CHN_Merging-1:     mergingBEVPredProb/{k}_{timestamp_idx}/T1.npy, T2.npy, T3.npy
  - DEU_Roundabout-1:  roundaboutBEVPredProb/{segment_id}_{timestamp_idx}/T1.npy, ...

All BEV maps: 288x288 float32 probability map covering 144m x 144m,
Resolution: 0.5 m/pixel. T1: +1s prediction, T2: +2s, T3: +3s.

Coordinate transformation (generic):
  INTERACTION = CR + (x_offset, y_offset)
  col_288 = 2 * ((INTERACTION.x - center_x) + 72.0 + 0.5)
  row_288 = 2 * (-(INTERACTION.y - center_y) + 72.0 + 0.5)

Scenario-specific geometry:
  USA_Intersection-1:  offset=(945.0, 993.5), center=(1003.0, 995.0)
  CHN_Merging-1:       offset=(1056.0, 954.5), center=(1075.0, 955.0)
  DEU_Roundabout-1:    offset=(897.0, 1004.0), center=(1000.0, 992.0)
"""

import os
import re
import numpy as np
from typing import Optional, Dict, Tuple, List

# ID offset for each recording k  (scenario_id = ID_OFFSET[k] + segment)
# k=0: 20 segments → IDs 1-20,   k=1: 18 segments → IDs 21-38, ...
ID_OFFSET = [1, 21, 39, 59, 79, 99, 119, 139]
NUM_SEGMENTS = [20, 18, 20, 20, 20, 20, 20, 17]  # segments per recording

# Scenario families supported by this loader
SCENARIO_USA_INTERSECTION = "USA_Intersection-1"
SCENARIO_CHN_MERGING = "CHN_Merging-1"
SCENARIO_DEU_ROUNDABOUT = "DEU_Roundabout-1"

# CHN_Merging mapping (derived from existing mergingBEVPredProb folders):
#   folder: 1_{timestamp_idx}, timestamp_idx in [69, 3499], step=10
#   scenario_id -> base timestamp_idx: 10 * scenario_id + 59
CHN_MERGING_RECORDING_K = 1
CHN_MERGING_BASE_OFFSET = 59
CHN_MERGING_SCENARIO_STRIDE = 10

# CHN_Merging BEV center (config item)
# Updated from automatic alignment search in this workspace.
CHN_MERGING_BEV_CENTER_X = 1075.0
CHN_MERGING_BEV_CENTER_Y = 955.0

# DEU_Roundabout: 12 MTSDB segments (1-12), each with BEV folders at stride 10.
# Timestamps are MTSDB-internal step indices (~100ms per unit).
# (min_ts, max_ts) per segment, derived from scanning roundaboutBEVPredProb/.
DEU_ROUNDABOUT_SEGMENT_RANGES = {
    1: (69, 3259), 2: (69, 3259), 3: (69, 3259), 4: (69, 3259),
    5: (109, 2009), 6: (69, 3259), 7: (69, 3259), 8: (69, 1469),
    9: (69, 3249), 10: (69, 2009), 11: (69, 2219), 12: (69, 2239),
}
DEU_ROUNDABOUT_SEGMENT_COUNT = 12
DEU_ROUNDABOUT_TOTAL_SCENARIOS = 210  # nominal max scenario ID
DEU_ROUNDABOUT_TS_STRIDE = 10  # BEV folders sampled at this index stride


def usa_scenario_id_to_recording(scenario_id: int) -> Tuple[int, int]:
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


def parse_benchmark_id(benchmark_id: str) -> Tuple[str, int]:
    """
    Extract (scenario family, scenario_id) from benchmark id.

    Supported examples:
      - USA_Intersection-1_3_T-1
      - CHN_Merging-1_57_T-1
    """
    m = re.search(r'(USA_Intersection-1|CHN_Merging-1|DEU_Roundabout-1)_(\d+)_T-1', benchmark_id)
    if m:
        return m.group(1), int(m.group(2))
    raise ValueError(f"Cannot parse supported benchmark_id from '{benchmark_id}'")


def usa_scenario_time_to_bev_timestamp(k: int, segment: int, time_step: int) -> int:
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

    # BEV grid parameters
    BEV_AREA_RANGE = 144.0   # meters covered
    BEV_SIZE = 288            # pixels
    BEV_RESOLUTION = BEV_AREA_RANGE / BEV_SIZE  # 0.5 m/pixel

    # Scenario-specific geometry config (built-in defaults).
    # Fallback used only if no external geometry is provided.
    SCENARIO_GEOMETRY = {
        SCENARIO_USA_INTERSECTION: {
            "x_offset": 945.0,
            "y_offset": 993.5,
            "center_x": 1003.0,
            "center_y": 995.0,
        },
        SCENARIO_CHN_MERGING: {
            "x_offset": 1056.0,
            "y_offset": 954.5,
            "center_x": CHN_MERGING_BEV_CENTER_X,
            "center_y": CHN_MERGING_BEV_CENTER_Y,
        },
        SCENARIO_DEU_ROUNDABOUT: {
            "x_offset": 897.0,
            "y_offset": 1004.0,
            "center_x": 1000.0,
            "center_y": 992.0,
        },
    }

    def __init__(
        self,
        bev_prob_dir: str,
        benchmark_id: str,
        bev_weight: float = 1.0,
        scenario_geometry: Optional[Dict] = None,
    ):
        """
        Initialize the BEV probability loader.

        Parameters:
            bev_prob_dir: Path to BEVPredProb root directory
            benchmark_id: CommonRoad scenario benchmark_id (e.g. 'USA_Intersection-1_3_T-1')
            bev_weight: Scaling weight for BEV risk contribution
            scenario_geometry: Optional dict with per-scenario geometry config.
                If provided, overrides built-in SCENARIO_GEOMETRY.
                Expected format: {
                    "USA_Intersection-1": {"x_offset": ..., "y_offset": ..., "center_x": ..., "center_y": ...},
                    "CHN_Merging-1": {...},
                    ...
                }
        """
        self.bev_prob_dir = bev_prob_dir
        self.benchmark_id = benchmark_id
        self.bev_weight = bev_weight
        self.scenario_family: Optional[str] = None
        self.scenario_id: Optional[int] = None
        self.k = -1
        self.segment = -1

        # Use external geometry if provided, otherwise fall back to built-in defaults.
        self._geometry_config = scenario_geometry if scenario_geometry else self.SCENARIO_GEOMETRY

        # Parse scenario ID and determine mapping mode
        try:
            self.scenario_family, self.scenario_id = parse_benchmark_id(benchmark_id)

            if self.scenario_family == SCENARIO_USA_INTERSECTION:
                self.k, self.segment = usa_scenario_id_to_recording(self.scenario_id)
            elif self.scenario_family == SCENARIO_CHN_MERGING:
                self.k = CHN_MERGING_RECORDING_K
                self.segment = self.scenario_id
            elif self.scenario_family == SCENARIO_DEU_ROUNDABOUT:
                # Determine MTSDB segment (1-12) from scenario_id.
                # Default: evenly distribute 210 nominal scenario IDs across 12 segments.
                scenarios_per_seg = DEU_ROUNDABOUT_TOTAL_SCENARIOS / DEU_ROUNDABOUT_SEGMENT_COUNT
                self.k = int((self.scenario_id - 1) // scenarios_per_seg) + 1
                self.k = max(1, min(DEU_ROUNDABOUT_SEGMENT_COUNT, self.k))
                self.segment = self.scenario_id  # store for reference
                # Default base_ts: segment's min timestamp.
                # This aligns scenario time_step 0 with the first BEV frame of the segment.
                self._roundabout_base_ts = DEU_ROUNDABOUT_SEGMENT_RANGES[self.k][0]
                self._roundabout_max_ts = DEU_ROUNDABOUT_SEGMENT_RANGES[self.k][1]
            else:
                raise ValueError(f"Unsupported scenario family: {self.scenario_family}")
        except ValueError:
            print(
                f"[BEVProbLoader] Warning: benchmark_id '{benchmark_id}' is not supported "
                f"(supported: {SCENARIO_USA_INTERSECTION}, {SCENARIO_CHN_MERGING}, "
                f"{SCENARIO_DEU_ROUNDABOUT}). BEV risk disabled."
            )
            self.available_timestamps = {}
            return

        # Select geometry by scenario family.
        # First check external/config-provided geometry, then built-in defaults.
        geom = self._geometry_config.get(self.scenario_family)
        if geom is None:
            geom = self.SCENARIO_GEOMETRY.get(
                self.scenario_family,
                self.SCENARIO_GEOMETRY[SCENARIO_USA_INTERSECTION],
            )

        # Keep uppercase aliases for backward compatibility with existing call sites.
        self.X_OFFSET = float(geom["x_offset"])
        self.Y_OFFSET = float(geom["y_offset"])
        self.BEV_CENTER_X = float(geom["center_x"])
        self.BEV_CENTER_Y = float(geom["center_y"])

        # Scan available BEV folders for this recording and segment
        self.available_timestamps = self._scan_available_timestamps()

        # Cache for loaded probability maps
        self._cached_ts: Optional[int] = None
        self._cached_maps: Optional[List[np.ndarray]] = None  # [T1, T2, T3]

        print(
            f"[BEVProbLoader] scenario={benchmark_id}, family={self.scenario_family}, "
            f"k={self.k}, segment={self.segment}, center=({self.BEV_CENTER_X:.1f}, "
            f"{self.BEV_CENTER_Y:.1f}), available BEV timestamps: "
            f"{len(self.available_timestamps)}"
        )

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

        # Scenario-specific timestamp mapping function:
        # key = scenario-internal time step, value = folder path.
        if self.scenario_family == SCENARIO_USA_INTERSECTION:
            # CR step range for this scenario
            cr_start = self.segment * 150 + 1   # first CR step of scenario
            cr_end = (self.segment + 1) * 150   # last CR step of scenario
        elif self.scenario_family == SCENARIO_CHN_MERGING:
            # For CHN_Merging: scenario start timestamp index in folder naming
            #   base_ts = 10 * scenario_id + 59
            chn_base_ts = (
                CHN_MERGING_SCENARIO_STRIDE * int(self.scenario_id)
                + CHN_MERGING_BASE_OFFSET
            )
        elif self.scenario_family == SCENARIO_DEU_ROUNDABOUT:
            # DEU_Roundabout: folders named {segment_id}_{timestamp_idx}.
            # timestamp_idx is in MTSDB step units (~100ms).
            # Map: scenario_ts ≈ timestamp_idx - base_ts (both in 100ms units).
            deu_base_ts = self._roundabout_base_ts
        else:
            return {}

        for folder_name in os.listdir(self.bev_prob_dir):
            if not folder_name.startswith(prefix):
                continue
            try:
                timestamp_ms = int(folder_name[len(prefix):])
            except ValueError:
                continue

            folder_path = os.path.join(self.bev_prob_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            if self.scenario_family == SCENARIO_USA_INTERSECTION:
                cr_step = timestamp_ms // 100
                if cr_start <= cr_step <= cr_end:
                    scenario_ts = cr_step - cr_start  # 0-based scenario time step
                    ts_map[scenario_ts] = folder_path
            elif self.scenario_family == SCENARIO_CHN_MERGING:
                # timestamp_ms here is actually an index in folder naming (e.g. 69, 79, ...).
                scenario_ts = timestamp_ms - chn_base_ts
                if scenario_ts >= 0:
                    ts_map[scenario_ts] = folder_path
            elif self.scenario_family == SCENARIO_DEU_ROUNDABOUT:
                # timestamp_ms is the MTSDB step index.
                # Map to scenario-relative time step (both in 100ms units).
                scenario_ts = timestamp_ms - deu_base_ts
                if scenario_ts >= 0:
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
            Maximum BEV probability along the trajectory, scaled by
            bev_weight.  Using max instead of sum avoids biasing toward
            shorter (faster) trajectories and keeps the value in [0, 1].
        """
        if self._cached_maps is None:
            return 0.0

        max_prob = 0.0
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
                # Beyond T3 prediction range, use T3
                t_idx = 2

            prob = self.query_prob(x, y, t_idx)
            if prob > max_prob:
                max_prob = prob

        return max_prob * self.bev_weight

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
            Maximum BEV probability in the segment, scaled by bev_weight.
        """
        if self._cached_maps is None:
            return 0.0

        max_prob = 0.0
        for point in traj_segment:
            x, y = point[0], point[1]
            prob = self.query_prob(x, y, t_index)
            if prob > max_prob:
                max_prob = prob

        return max_prob * self.bev_weight

    @property
    def is_available(self) -> bool:
        """Check if BEV probability data is available for this scenario."""
        return self.k >= 0 and len(self.available_timestamps) > 0
