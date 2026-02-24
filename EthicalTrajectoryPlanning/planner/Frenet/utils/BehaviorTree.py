#!/user/bin/env python

"""Sampling-based trajectory planning in a frenet frame considering ethical implications."""
from typing import List

# Third party imports
from commonroad.scenario.trajectory import State
class LateralBehavior:
    def target_offset(self, root_offset, state: State):
        offset = 0
        return offset


class LateralKeep(LateralBehavior):
    def target_offset(self, root_offset, state: State):
        offset = root_offset
        return offset


class LateralLeft(LateralBehavior):
    def target_offset(self, root_offset, state: State):
        # offset = root_offset - min(state.velocity / 20, 1.0)

        offset = root_offset
        if state.velocity < 5:
            offset -= 0.3
        else:
            offset -= 0.5
        return offset


class LateralRight(LateralBehavior):
    def target_offset(self, root_offset, state: State):
        # offset = root_offset + min(state.velocity / 20, 1.0)

        offset = root_offset
        if state.velocity < 5:
            offset += 0.3
        else:
            offset += 0.5
        return offset


class LongitudinalBehavior:
    def target_velocity(self, root_velocity, state: State):
        velocity = state.velocity
        return velocity


class LongitudinalCruise(LongitudinalBehavior):
    def target_velocity(self, root_velocity, state: State):
        velocity = state.velocity
        return velocity

class SoftAcceleration(LongitudinalBehavior):
    def target_velocity(self, root_velocity, state: State):
        velocity = min(state.velocity + 2.5, 50)
        return velocity


class HardAcceleration(LongitudinalBehavior):
    def target_velocity(self, root_velocity, state: State):
        velocity = min(state.velocity + 5, 50)
        return velocity


class SoftDeceleration(LongitudinalBehavior):
    def target_velocity(self, root_velocity, state: State):
        velocity = max(state.velocity - 3, 0.01)
        return velocity


class HardDeceleration(LongitudinalBehavior):
    def target_velocity(self, root_velocity, state: State):
        velocity = max(state.velocity - 10, 0.01)
        return velocity

class BehaviorNode:
    def __init__(self, lateral_behavior, longitudinal_behavior):

        self.behavior = (lateral_behavior, longitudinal_behavior)
        self.children = []
        self.children_behavior = []
    # def target_behavior(self, root_offset, root_velocity, state: State):
    #     return self.offset_func(root_offset, state), self.velocity_func(root_velocity, state)

    def add_child(self, node):
        self.children.append(node)
        self.children_behavior.append(node.behavior)

    def find(self, behavior):
        if behavior in self.children_behavior:
            index = self.children_behavior.index(behavior)
            return self.children[index]
        else:
            return None

class BehaviorTree:
    def __init__(self):
        self.root = BehaviorNode(-1, -1)
        self.lateral_behaviors_dict = {"Keep": LateralKeep(), "Left": LateralLeft(), "Right": LateralRight()}
        self.longitudinal_behaviors_dict = {"Cruise": LongitudinalCruise(), "SoftAcc": SoftAcceleration(),
                                       "HardAcc": HardAcceleration(), "SoftDec": SoftDeceleration(),
                                       "HardDec": HardDeceleration()}
        # keep for 1s, then change for 2s, 12 paths
        self.add_path([BehaviorNode("Left", "Cruise"),
                       BehaviorNode("Keep", "SoftAcc"),
                       BehaviorNode("Keep", "SoftAcc")])
        self.add_path([BehaviorNode("Right", "Cruise"),
                       BehaviorNode("Keep", "SoftAcc"),
                       BehaviorNode("Keep", "SoftAcc")])
        # self.add_path([BehaviorNode("Keep", "Cruise"),
        #                BehaviorNode("Left", "Cruise"),
        #                BehaviorNode("Left", "Cruise")])
        # self.add_path([BehaviorNode("Keep", "SoftDec"),
        #                BehaviorNode("Left", "Cruise"),
        #                BehaviorNode("Left", "Cruise")])
        self.add_path([BehaviorNode("Left", "Cruise"),
                       BehaviorNode("Left", "SoftAcc"),
                       BehaviorNode("Keep", "SoftAcc")])
        self.add_path([BehaviorNode("Right", "Cruise"),
                       BehaviorNode("Right", "SoftAcc"),
                       BehaviorNode("Keep", "SoftAcc")])
        # self.add_path([BehaviorNode("Keep", "SoftDec"),
        #                BehaviorNode("Left", "Cruise"),
        #                BehaviorNode("Right", "Cruise")])


        # self.add_path([BehaviorNode("Keep", "Cruise"),
        #                BehaviorNode("Right", "Cruise"),
        #                BehaviorNode("Right", "Cruise")])
        # self.add_path([BehaviorNode("Keep", "SoftDec"),
        #                BehaviorNode("Right", "Cruise"),
        #                BehaviorNode("Right", "Cruise")])
        # self.add_path([BehaviorNode("Keep", "SoftAcc"),
        #                BehaviorNode("Right", "Cruise"),
        #                BehaviorNode("Left", "Cruise")])
        # self.add_path([BehaviorNode("Keep", "Cruise"),
        #                BehaviorNode("Right", "Cruise"),
        #                BehaviorNode("Left", "Cruise")])
        # self.add_path([BehaviorNode("Keep", "SoftDec"),
        #                BehaviorNode("Right", "Cruise"),
        #                BehaviorNode("Left", "Cruise")])

        # change for 2s, then keep, 4 paths
        # self.add_path([BehaviorNode("Left", "SoftAcc"),
        #                BehaviorNode("Right", "SoftAcc"),
        #                BehaviorNode("Keep", "SoftAcc")])
        # self.add_path([BehaviorNode("Left", "SoftAcc"),
        #                BehaviorNode("Right", "SoftAcc"),
        #                BehaviorNode("Keep", "Cruise")])
        # self.add_path([BehaviorNode("Right", "SoftAcc"),
        #                BehaviorNode("Left", "SoftAcc"),
        #                BehaviorNode("Keep", "SoftAcc")])
        # self.add_path([BehaviorNode("Right", "SoftAcc"),
        #                BehaviorNode("Left", "SoftAcc"),
        #                BehaviorNode("Keep", "Cruise")])

        # change for 3s, keep in the 2nd second, 4 paths
        # self.add_path([BehaviorNode("Left", "SoftAcc"),
        #                BehaviorNode("Keep", "Cruise"),
        #                BehaviorNode("Right", "Cruise")])
        self.add_path([BehaviorNode("Keep", "Cruise"),
                       BehaviorNode("Keep", "Cruise"),
                       BehaviorNode("Left", "SoftAcc")])
        # self.add_path([BehaviorNode("Right", "SoftAcc"),
        #                BehaviorNode("Keep", "Cruise"),
        #                BehaviorNode("Left", "Cruise")])
        self.add_path([BehaviorNode("Keep", "Cruise"),
                       BehaviorNode("Keep", "Cruise"),
                       BehaviorNode("Right", "SoftAcc")])

        # keep, keep same longitudinal behavior, 5 paths
        self.add_path([BehaviorNode("Keep", "Cruise"),
                       BehaviorNode("Keep", "Cruise"),
                       BehaviorNode("Keep", "Cruise")])
        self.add_path([BehaviorNode("Keep", "SoftAcc"),
                       BehaviorNode("Keep", "SoftAcc"),
                       BehaviorNode("Keep", "SoftAcc")])
        self.add_path([BehaviorNode("Keep", "SoftDec"),
                       BehaviorNode("Keep", "SoftDec"),
                       BehaviorNode("Keep", "SoftDec")])
        # self.add_path([BehaviorNode("Keep", "HardAcc"),
        #                BehaviorNode("Keep", "HardAcc"),
        #                BehaviorNode("Keep", "HardAcc")])
        # self.add_path([BehaviorNode("Keep", "HardDec"),
        #                BehaviorNode("Keep", "HardDec"),
        #                BehaviorNode("Keep", "HardDec")])

        # one longitudinal behavior for 1s, another for 2s, 6 paths
        self.add_path([BehaviorNode("Keep", "Cruise"),
                       BehaviorNode("Keep", "SoftDec"),
                       BehaviorNode("Keep", "SoftDec")])

        self.add_path([BehaviorNode("Keep", "SoftAcc"),
                       BehaviorNode("Keep", "Cruise"),
                       BehaviorNode("Keep", "Cruise")])
        # self.add_path([BehaviorNode("Keep", "SoftDec"),
        #                BehaviorNode("Keep", "Cruise"),
        #                BehaviorNode("Keep", "Cruise")])
        self.add_path([BehaviorNode("Keep", "HardDec"),
                       BehaviorNode("Keep", "Cruise"),
                       BehaviorNode("Keep", "Cruise")])
        # self.add_path([BehaviorNode("Keep", "HardAcc"),
        #                BehaviorNode("Keep", "Cruise"),
        #                BehaviorNode("Keep", "Cruise")])
        self.add_path([BehaviorNode("Keep", "HardDec"),
                       BehaviorNode("Keep", "SoftAcc"),
                       BehaviorNode("Keep", "SoftAcc")])
        # self.add_path([BehaviorNode("Keep", "HardAcc"),
        #                BehaviorNode("Keep", "SoftAcc"),
        #                BehaviorNode("Keep", "SoftAcc")])

        # one longitudinal behavior for 2s, another for 1s, 6 paths
        # self.add_path([BehaviorNode("Keep", "SoftDec"),
        #                BehaviorNode("Keep", "SoftDec"),
        #                BehaviorNode("Keep", "Cruise")])
        # self.add_path([BehaviorNode("Keep", "SoftAcc"),
        #                BehaviorNode("Keep", "SoftAcc"),
        #                BehaviorNode("Keep", "Cruise")])
        # self.add_path([BehaviorNode("Keep", "Cruise"),
        #                BehaviorNode("Keep", "Cruise"),
        #                BehaviorNode("Keep", "SoftAcc")])
        # self.add_path([BehaviorNode("Keep", "Cruise"),
        #                BehaviorNode("Keep", "Cruise"),
        #                BehaviorNode("Keep", "SoftDec")])

        self.add_path([BehaviorNode("Keep", "HardDec"),
                       BehaviorNode("Keep", "HardDec"),
                       BehaviorNode("Keep", "SoftAcc")])


        # three different longitudinal behaviors, 8 paths
        # self.add_path([BehaviorNode("Keep", "HardDec"),
        #                BehaviorNode("Keep", "SoftAcc"),
        #                BehaviorNode("Keep", "HardAcc")])
        # self.add_path([BehaviorNode("Keep", "HardAcc"),
        #                BehaviorNode("Keep", "Cruise"),
        #                BehaviorNode("Keep", "HardDec")])
        # self.add_path([BehaviorNode("Keep", "HardAcc"),
        #                BehaviorNode("Keep", "Cruise"),
        #                BehaviorNode("Keep", "SoftDec")])
        self.add_path([BehaviorNode("Keep", "SoftAcc"),
                       BehaviorNode("Keep", "Cruise"),
                       BehaviorNode("Keep", "SoftDec")])
        self.add_path([BehaviorNode("Keep", "Cruise"),
                       BehaviorNode("Keep", "SoftDec"),
                       BehaviorNode("Keep", "HardDec")])
        self.add_path([BehaviorNode("Keep", "Cruise"),
                       BehaviorNode("Keep", "SoftAcc"),
                       BehaviorNode("Keep", "HardAcc")])
        # self.add_path([BehaviorNode("Keep", "HardDec"),
        #                BehaviorNode("Keep", "SoftDec"),
        #                BehaviorNode("Keep", "Cruise")])
        # self.add_path([BehaviorNode("Keep", "HardAcc"),
        #                BehaviorNode("Keep", "SoftAcc"),
        #                BehaviorNode("Keep", "Cruise")])

        # 1-0-1, 2 paths
        # self.add_path([BehaviorNode("Keep", "SoftAcc"),
        #                BehaviorNode("Keep", "Cruise"),
        #                BehaviorNode("Keep", "SoftAcc")])
        # self.add_path([BehaviorNode("Keep", "SoftAcc"),
        #                BehaviorNode("Keep", "SoftDec"),
        #                BehaviorNode("Keep", "SoftAcc")])
        print("hello")

    def get_behaviors_in_path(self, parent_action, state, paths_behavior):
        search_node = self.root
        behavior_list = []
        action_list = []
        parent_d, parent_v = parent_action
        for behavior in paths_behavior:
            search_node = search_node.find(behavior)
        if search_node is None:
            print("he;")
        for (lateral_behavior, longitudinal_behavior) in search_node.children_behavior:
            d, v = (self.lateral_behaviors_dict[lateral_behavior].target_offset(parent_d, state),
                    self.longitudinal_behaviors_dict[longitudinal_behavior].target_velocity(parent_v, state))
            behavior_list.append((lateral_behavior, longitudinal_behavior))
            action_list.append((d, v))
        return behavior_list, action_list


    def add_path(self, behavior_path: List[BehaviorNode]):
        search_node = self.root
        for node in behavior_path:
            child_node = search_node.find(node.behavior)
            if child_node is None:
                search_node.add_child(node)
                search_node = node
            else:
                search_node = child_node

    def print_paths(self):
        res = []
        self.recursive_path(self.root, res, "")
        for path in res:
            print(path)

    def recursive_path(self, node: BehaviorNode, res, path_str):
        if len(node.children) == 0:
            res.append(path_str)
        else:
            for i, child in enumerate(node.children):
                self.recursive_path(child, res, path_str + str(child.behavior) + "->")

if __name__ == '__main__':
    Tree = BehaviorTree()
    Tree.print_paths()