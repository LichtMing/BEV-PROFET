import os
import csv
import numpy as np
from commonroad.scenario.scenario import Scenario
from scipy.integrate import simps
from commonroad_helper_functions.spacial import get_leader_on_lanelet
def find_best_traj(valid_trajs,
                   traj_risk):

    K = len(valid_trajs) if len(valid_trajs) < 10 else 10
    Nmin_risk_lis, Nmin_idx_lis, Nmin_traj_lis = [], [], []

    max_risk = max(traj_risk)
    max_idx = traj_risk.index(max_risk)
    max_risk_traj = valid_trajs[max_idx]

    for _ in range(K):
        min_val = min(traj_risk)
        min_idx = traj_risk.index(min_val)

        Nmin_risk_lis.append(min_val)
        Nmin_idx_lis.append(min_idx)
        Nmin_traj_lis.append(valid_trajs[min_idx])

        traj_risk[min_idx] = float('inf')

    metric_list = []
    for traj in Nmin_traj_lis:
        metric_list.append(traj_metric(traj))

    metric_list = np.asarray(metric_list)
    for i in range(len(metric_list[0])):
        metric_list[:, i] /= max(metric_list[:, i])

    # weighted_metric = lambda m: m[0] * 0.6 - m[1] * 0.1 + m[2] * 0.1 - m[3] * 0.4
    # weighted_metric = lambda m: m[0]
    weighted_metric = lambda m: m[0] * 0.5 - m[1] * 0.5 + m[2] * 0.1 - m[3] * 0.1
    metric_vals = [weighted_metric(m) for m in metric_list]
    best_traj_index = metric_vals.index(max(metric_vals))
    best_traj = Nmin_traj_lis[best_traj_index]
    best_traj_risk = Nmin_risk_lis[best_traj_index]
    return best_traj, best_traj_risk, max_risk_traj, max_risk

def traj_metric(traj):
    s_traj = abs(traj[-1][2] - traj[0][2])
    d_sum = 0
    v_list = [s[4] for s in traj]
    time_gap = 3.0 / (len(v_list) - 1)
    acceleration = np.diff(v_list) / time_gap
    acceleration_sq = np.square(acceleration)
    a_sq_inter = simps(acceleration_sq, dx=time_gap)
    for sample_point in traj:
        d_sum += abs(sample_point[3])

    v_avg = np.mean(v_list)
    return s_traj, d_sum, v_avg, a_sq_inter

def calc_eval_values(scenario: Scenario, ego_id, time_step, traj, reference_spline):
    ttc_list = []
    thw_list = []
    ttc = -1
    thw = -1
    ego_obs = scenario.obstacle_by_id(ego_id)
    max_ts = None
    if ego_obs is not None and ego_obs.prediction is not None:
        max_ts = ego_obs.prediction.trajectory.state_list[-1].time_step

    for i, state in enumerate(traj):
        if i % 5 != 0:
            continue
        if (
                len(
                    scenario.lanelet_network.find_lanelet_by_position(
                        [np.asarray([state[0], state[1]])]
                    )[0]
                )
                > 0
        ):
            lanelet_id = (
                scenario.lanelet_network.find_lanelet_by_position(
                    [np.asarray([state[0], state[1]])]
                )[0][0]
            )
        else:
            continue

        query_time = time_step + i + 1
        if max_ts is not None:
            query_time = min(query_time, max_ts)
        leader_id, distance, approaching_rate = get_leader_on_lanelet(scenario, ego_id, lanelet_id, query_time)
        if leader_id is None:
            continue
        ttc = distance / (approaching_rate + 0.01)
        thw = distance / (state[4] + 0.01)
        ttc_list.append(ttc)
        thw_list.append(thw)

    if len(ttc_list) > 0:
        ttc_list = np.clip(ttc_list, -10, 10)
        ttc = float(np.mean(ttc_list))
    if len(thw_list) > 0:
        thw_list = np.clip(thw_list, 0, 30)
        thw = float(np.mean(thw_list))
    return {"ttc": ttc, "thw": thw}

class TrajLogger(object):
    def __init__(self, log_prefix, points_num, save_dir=None):
        self.db_s, self.db_v, self.db_a, self.db_r = [], [], [], []
        self.db_t_pred, self.db_t_planning, self.db_t_evaluation = [], [], []
        self.db_ttc, self.db_thw = [], []
        self.db_ego_id, self.db_time_step = [], []
        self.s_avg, self.v_avg, self.a_avg, self.r_avg = 0, 0, 0, 0
        self.t_pred_avg, self.t_planning_avg, self.t_evaluation_avg = 0, 0, 0
        self.ttc_avg, self.thw_avg = -1, -1
        self.log_prefix = log_prefix
        self.points_num = points_num
        self.save_dir = save_dir
        self.db_length = 0

    def record_traj(self, ego_id, time_step, traj, risk, eval_res, duration_pred, duration_planning, duration_evaluation):
        s = self.path_length(traj)
        v = self.average_velocity(traj)
        a = self.acc_cost(traj)
        self.db_s.append(s)
        self.db_v.append(v)
        self.db_a.append(a)
        self.db_r.append(risk)
        self.db_ego_id.append(ego_id)
        self.db_time_step.append(time_step)
        self.db_t_pred.append(duration_pred)
        self.db_t_planning.append(duration_planning)
        self.db_t_evaluation.append(duration_evaluation)
        if eval_res["ttc"] != -1:
            self.db_ttc.append(eval_res["ttc"])
        if eval_res["thw"] != -1:
            self.db_thw.append(eval_res["thw"])
        self.db_length += 1
        print("vehicle {} at {}'s traj from {} : s {}, v {}, a {}, r {}, ttc {}, thw {}".format(ego_id, time_step, self.log_prefix,
                                                                          s, v, a, risk, eval_res["ttc"], eval_res["thw"]))

    def log_average_traj_data(self, save_dir=None):
        self.s_avg = np.mean(self.db_s)
        self.v_avg = np.mean(self.db_v)
        self.a_avg = np.mean(self.db_a)
        self.r_avg = np.mean(self.db_r)
        self.t_pred_avg = np.mean(self.db_t_pred)
        self.t_planning_avg = np.mean(self.db_t_planning)
        self.t_evaluation_avg = np.mean(self.db_t_evaluation)
        self.ttc_avg = np.mean(self.db_ttc) if len(self.db_ttc) > 0 else -1
        self.thw_avg = np.mean(self.db_thw) if len(self.db_thw) > 0 else -1
        print(self.points_num)
        print("trajs from {} have the average data: s {}, v {}, a {}, r {}, r for point {}, t_pred {}, t_plan {}, t_eval {},  ttc {}, thw {}".format(self.log_prefix,
                                                                          self.s_avg, self.v_avg, self.a_avg, self.r_avg, self.r_avg / self.points_num,
                                                                            self.t_pred_avg, self.t_planning_avg, self.t_evaluation_avg,
                                                                              self.ttc_avg, self.thw_avg))
        out_dir = save_dir or self.save_dir
        if out_dir is not None:
            self.save_to_csv(out_dir)

    def extend_logger(self, logger):
        self.db_s.extend(logger.db_s)
        self.db_a.extend(logger.db_a)
        self.db_v.extend(logger.db_v)
        self.db_r.extend(logger.db_r)
        self.db_ego_id.extend(logger.db_ego_id)
        self.db_time_step.extend(logger.db_time_step)
        self.db_t_pred.extend(logger.db_t_pred)
        self.db_t_planning.extend(logger.db_t_planning)
        self.db_t_evaluation.extend(logger.db_t_evaluation)
        self.db_ttc.extend(logger.db_ttc)
        self.db_thw.extend(logger.db_thw)

    def path_length(self, traj):
        return traj[-1][2] - traj[0][2]

    def acc_cost(self, traj):
        v_list = [s[4] for s in traj]
        time_gap = 3 / (len(v_list) - 1)
        acceleration = np.diff(v_list) / time_gap
        acceleration_sq = np.square(acceleration)
        a_sq_inter = simps(acceleration_sq, dx=time_gap)
        return a_sq_inter

    def average_velocity(self, traj):
        return np.mean([s[4] for s in traj])

    def save_to_csv(self, save_dir):
        """Save per-step records and average summary to CSV files in save_dir."""
        os.makedirs(save_dir, exist_ok=True)
        tag = self.log_prefix.replace(" ", "_")

        # --- per-step detail CSV ---
        detail_path = os.path.join(save_dir, f"eval_detail_{tag}.csv")
        with open(detail_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ego_id", "time_step", "travel_distance", "avg_velocity",
                             "acc_cost", "cumulative_risk", "t_pred", "t_planning", "t_evaluation"])
            for i in range(len(self.db_s)):
                writer.writerow([
                    self.db_ego_id[i] if i < len(self.db_ego_id) else "",
                    self.db_time_step[i] if i < len(self.db_time_step) else "",
                    self.db_s[i], self.db_v[i], self.db_a[i], self.db_r[i],
                    self.db_t_pred[i], self.db_t_planning[i], self.db_t_evaluation[i],
                ])

        # --- average summary CSV ---
        summary_path = os.path.join(save_dir, f"eval_summary_{tag}.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["avg_travel_distance", self.s_avg])
            writer.writerow(["avg_velocity", self.v_avg])
            writer.writerow(["avg_acc_cost", self.a_avg])
            writer.writerow(["avg_cumulative_risk", self.r_avg])
            writer.writerow(["avg_risk_per_point", self.r_avg / self.points_num if self.points_num else 0])
            writer.writerow(["avg_ttc", self.ttc_avg])
            writer.writerow(["avg_thw", self.thw_avg])
            writer.writerow(["avg_t_pred", self.t_pred_avg])
            writer.writerow(["avg_t_planning", self.t_planning_avg])
            writer.writerow(["avg_t_evaluation", self.t_evaluation_avg])
            writer.writerow(["num_records", len(self.db_s)])
        print(f"Evaluation saved to {detail_path} and {summary_path}")