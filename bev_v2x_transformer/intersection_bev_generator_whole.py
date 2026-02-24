import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import DBtools.utils as utils
from DBtools.init_db import init_DB
import numpy as np
import cv2
import scipy.io
from PIL import Image
from scipy import ndimage
from InsertDataBase.Interaction_Intersection_EP0_InsertParticipant import ProcessPolygon, inPoly
from scipy.stats import beta
from datetime import datetime
import os
import torch
from torchvision import datasets, transforms


def mask_grid_from_lane_pos(x_list, y_list, center_x, center_y, map_size, ratio):
    x_list = np.array(x_list).reshape(-1, 1).astype(np.float)
    y_list = np.array(y_list).reshape(-1, 1).astype(np.float)
    pts_ori = np.hstack((x_list, y_list))
    pts_ori[:, 0] = (pts_ori[:, 0] - center_x) / ratio + map_size / 2 + 0.5
    pts_ori[:, 1] = - (pts_ori[:, 1] - center_y) / ratio + map_size / 2 + 0.5
    pts_ori = pts_ori.astype(np.int)
    pts_ori = list(pts_ori.tolist())
    for idx in range(len(pts_ori)):
        pts_ori[idx] = (pts_ori[idx][0], pts_ori[idx][1])
    # For grid occupancy remove duplicates
    pts = list(set(pts_ori))
    pts.sort(key=pts_ori.index)
    pts = np.array(pts)
    minus = np.sum(pts < 0)
    plus = np.sum(pts >= map_size)
    if minus > 0 or plus > 0:
        return pts, False
    else:
        return pts, True


def draw_dashed_line(img, pt1, pt2, color, thickness, style, gap):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts_all = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts_all.append(p)

    if style == 'dotted':
        for p in pts_all:
            cv2.circle(img, p, thickness, color, -1)
    else:
        pre = pts_all[0]
        next = pts_all[0]
        i = 0
        for p in pts_all:
            pre = next
            next = p
            if i % 2 == 1:
                cv2.line(img, pre, next, color, thickness)
            i = i + 1


def draw_poly_dashed_line(mask, pts, color, thickness, style, gap):
    for i in range(pts.shape[0] - 1):
        draw_dashed_line(mask, pts[i], pts[i + 1], color, thickness, style, gap)


def draw_drivable_area(cursor, mask, center_x, center_y, map_size, ratio):
    polygon_list, _ = ProcessPolygon(cursor)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            local_x = (j - map_size / 2) * ratio + center_x
            local_y = center_y - (i - map_size / 2) * ratio
            for k in range(len(polygon_list)):
                if (inPoly(polygon_list[k][:, 1:], [local_x, local_y])):
                    cv2.circle(mask, (j, i), 1, (220, 220, 220), -1)
                    break
    return mask


def draw_map_part(cursor, mask, Xlist, Ylist, center_x, center_y, ratio):
    way_list = utils.SearchWayFromDB(cursor)
    # print(way_list)
    map_size = mask.shape[0]
    for way_id in way_list:
        node_list, x_list, y_list = utils.SearchNodeIDOnWayFromDB(cursor, way_id=way_id, x_range=Xlist, y_range=Ylist)
        pts, flag = mask_grid_from_lane_pos(x_list, y_list, center_x, center_y, map_size, ratio)
        if flag == False:
            continue

        way_type, way_subtype = utils.SearchWayTypeFromDB(cursor, way_id=way_id)
        if way_type == "curbstone": # (256, 265, 0)
            cv2.polylines(mask, [pts], False, (192, 192, 192), 2)
        elif way_type == "line_thin":
            if way_subtype == "dashed":
                draw_poly_dashed_line(mask, pts, (192, 192, 192), 1, "dashed", 2)
            else:
                draw_poly_dashed_line(mask, pts, (192, 192, 192), 1, "dashed", 2)
        else: # (192, 192, 192)
            draw_poly_dashed_line(mask, pts, (192, 192, 192), 1, "dashed", 2)

        channelization = utils.SearchChannelizationOnWayFromDB(cursor, way_id=way_id)

        for key in channelization:
            if key == "pedestrian_marking":
                draw_poly_dashed_line(mask, pts, (192, 192, 192), 1, "dashed", 2)
            elif key == "stop_line":
                draw_poly_dashed_line(mask, pts, (192, 192, 192), 1, "dashed", 2)
            elif key == "road_border":
                draw_poly_dashed_line(mask, pts, (192, 192, 192), 1, "dashed", 2)
            elif key == "guard_rail":
                draw_poly_dashed_line(mask, pts, (192, 192, 192), 1, "dashed", 2)
            elif key == "turn_direction":
                draw_poly_dashed_line(mask, pts, (192, 192, 192), 1, "dashed", 2)
                break
            elif key == "is_intersection":
                draw_poly_dashed_line(mask, pts, (192, 192, 192), 1, "dashed", 2)
                break

    return mask


def mask_grid_from_vehicle_pos(position_vector, center_x, center_y, area_range, ratio):
    map_size = int(area_range / ratio)
    position_vector[:, 0] = (position_vector[:, 0] - center_x) / ratio + map_size / 2 + 0.5
    position_vector[:, 1] = - (position_vector[:, 1] - center_y) / ratio + map_size / 2 + 0.5
    position_vector = position_vector.astype(np.int)
    minus = np.sum(position_vector < 0)
    plus = np.sum(position_vector >= map_size)
    if minus > 0 or plus > 0:
        return position_vector, False
    else:
        return position_vector, True


def rotate_around_center(pts, center, yaw):
    return np.dot(pts - center, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])) + center


def polygon_xy_from_motionstate(x, y, psi_rad, width, length):
    lowleft = (x - length / 2., y - width / 2.)
    lowright = (x + length / 2., y - width / 2.)
    upright = (x + length / 2., y + width / 2.)
    upleft = (x - length / 2., y + width / 2.)
    return rotate_around_center(np.array([lowleft, lowright, upright, upleft]), np.array([x, y]), yaw=psi_rad)


def draw_vehicle_and_trajectory(cursor, mask, local_x, local_y, timestamp, table, single_range, ratio, vehicle_list):
    for vehicle in vehicle_list:
        x, y, orientation = utils.SearchTrafficParticipantTiming(cursor, vehicle, timestamp, table)
        vehicle_property = utils.SearchTrafficParticipantProperty(cursor, vehicle, table)
        min_dist = np.sqrt((x - local_x) * (x - local_x) + (y - local_y) * (y - local_y))
        if min_dist < single_range / 2:
            vehicle_length = vehicle_property["vehicle_length"]
            vehicle_width = vehicle_property["vehicle_width"]
            vehicle_position = polygon_xy_from_motionstate(x, y, orientation, vehicle_width, vehicle_length)
            vehicle_grid, flag = mask_grid_from_vehicle_pos(vehicle_position, local_x, local_y, single_range, ratio)
            if flag:
                cv2.fillPoly(mask, [vehicle_grid], (65, 105, 225))

    return mask


def draw_single_vehicle_and_trajectory(cursor, mask, ego, timestamp, table, area_range, ratio):
    local_x, local_y, orien = utils.SearchTrafficParticipantTiming(cursor, ego, timestamp, table)
    vehicle_list = utils.SearchEgoSameTimeVehicle(cursor, ego, timestamp, table)
    ego_property = utils.SearchTrafficParticipantProperty(cursor, ego, table)
    ego_location = [local_x, local_y, orien]
    vehicle_length = ego_property["vehicle_length"]
    vehicle_width = ego_property["vehicle_width"]
    ego_position = polygon_xy_from_motionstate(local_x, local_y, orien, vehicle_width, vehicle_length)
    ego_grid, flag = mask_grid_from_vehicle_pos(ego_position, local_x, local_y, area_range, ratio)
    ego_mask = None
    if flag:
        cv2.fillPoly(mask, [ego_grid], (60, 179, 113))
        ego_mask = mask.copy()

    for vehicle in vehicle_list:
        x, y, orientation = utils.SearchTrafficParticipantTiming(cursor, vehicle, timestamp, table)
        vehicle_location = [x, y, orientation]
        vehicle_property = utils.SearchTrafficParticipantProperty(cursor, vehicle, table)
        _, min_dist, _ = utils.min_distance_between_vehicles(ego_location, ego_property,
                                                             vehicle_location, vehicle_property)
        if min_dist < area_range / 2:
            vehicle_length = vehicle_property["vehicle_length"]
            vehicle_width = vehicle_property["vehicle_width"]
            vehicle_position = polygon_xy_from_motionstate(x, y, orientation, vehicle_width, vehicle_length)
            vehicle_grid, flag = mask_grid_from_vehicle_pos(vehicle_position, local_x, local_y, area_range, ratio)
            if flag:
                cv2.fillPoly(mask, [vehicle_grid], (65, 105, 225))

    return ego_mask, mask


def rotate_bev_img(img, radis):
    width = img.shape[0]
    height = img.shape[1]
    cos_radian = np.cos(radis)
    sin_radian = np.sin(radis)
    new_img = np.zeros(img.shape, dtype=np.uint8)
    dx = 0.5 * width + 0.5 * height * sin_radian - 0.5 * width * cos_radian
    dy = 0.5 * height - 0.5 * width * sin_radian - 0.5 * height * cos_radian
    # Forward Reflection
    for y0 in range(height):
        for x0 in range(width):
            x = x0 * cos_radian - y0 * sin_radian + dx
            y = x0 * sin_radian + y0 * cos_radian + dy
            if 0 < int(x) <= width and 0 < int(y) <= height:
                new_img[int(y) - 1, int(x) - 1] = img[int(y0), int(x0)]
    # Back Reflection
    dx_back = 0.5 * width - 0.5 * width * cos_radian - 0.5 * height * sin_radian
    dy_back = 0.5 * height + 0.5 * width * sin_radian - 0.5 * height * cos_radian
    for y in range(height):
        for x in range(width):
            x0 = x * cos_radian + y * sin_radian + dx_back
            y0 = y * cos_radian - x * sin_radian + dy_back
            if 0 < int(x0) <= width and 0 < int(y0) <= height:
                new_img[int(y), int(x)] = img[int(y0) - 1, int(x0) - 1]
    new_img[np.sum(new_img, 2) == 0] = [255, 255, 255]
    return new_img


def rotate_bev_label(img, radis):
    width = img.shape[0]
    height = img.shape[1]
    cos_radian = np.cos(radis)
    sin_radian = np.sin(radis)
    new_img = np.zeros(img.shape, dtype=np.float)
    dx = 0.5 * width + 0.5 * height * sin_radian - 0.5 * width * cos_radian
    dy = 0.5 * height - 0.5 * width * sin_radian - 0.5 * height * cos_radian
    # Forward Reflection
    for y0 in range(height):
        for x0 in range(width):
            x = x0 * cos_radian - y0 * sin_radian + dx
            y = x0 * sin_radian + y0 * cos_radian + dy
            if 0 < int(x) <= width and 0 < int(y) <= height:
                new_img[int(y) - 1, int(x) - 1] = img[int(y0), int(x0)]
    # Back Reflection
    dx_back = 0.5 * width - 0.5 * width * cos_radian - 0.5 * height * sin_radian
    dy_back = 0.5 * height + 0.5 * width * sin_radian - 0.5 * height * cos_radian
    for y in range(height):
        for x in range(width):
            x0 = x * cos_radian + y * sin_radian + dx_back
            y0 = y * cos_radian - x * sin_radian + dy_back
            if 0 < int(x0) <= width and 0 < int(y0) <= height:
                new_img[int(y), int(x)] = img[int(y0) - 1, int(x0) - 1]
    # new_img[np.sum(new_img, 2) == 0] = [255, 255, 255]
    return new_img


def ge_single_label_from_whole(cursor, veh, timestamp, table, area_range, single_range, mask_ori_label, ratio):
    min_x = 931
    min_y = 923
    single_map_size = int(single_range / ratio)
    area_map_size = int(area_range / ratio)
    local_x, local_y, orien = utils.SearchTrafficParticipantTiming(cursor, veh, timestamp, table)
    cx = int((local_x - min_x) / ratio + 0.5)
    cy = int(area_map_size - (local_y - min_y) / ratio + 0.5)
    mask_ori_trans = mask_ori_label.transpose(1, 2, 0)
    single_ori_label = np.zeros((single_map_size, single_map_size, 2), np.uint8)
    a = cx - single_map_size / 2
    b = cx + single_map_size / 2
    c = cy - single_map_size / 2
    d = cy + single_map_size / 2
    xl = 0
    yl = 0
    if a < 0:
        xl = abs(a)
    else:
        xl = 0
    if c < 0:
        yl = abs(c)
    else:
        yl = 0

    if a < 0:
        a = 0
    if b > area_map_size:
        b = area_map_size
    if c < 0:
        c = 0
    if d > area_map_size:
        d = area_map_size

    mask_inner = mask_ori_trans[int(c): int(d), int(a): int(b), :]
    single_ori_label[int(yl):int(mask_inner.shape[0] + yl), int(xl):int(mask_inner.shape[1] + xl), :] = mask_inner
    vehicle_mask = np.ones((single_map_size, single_map_size, 3), np.uint8) * 255
    ego_mask, vehicle_mask = draw_single_vehicle_and_trajectory(cursor, vehicle_mask, veh, timestamp, table, single_range, ratio)
    ego_label = (np.sum(ego_mask, 2) != 255 * 3).astype(np.uint8)
    vehicle_label = (np.sum(vehicle_mask, 2) != 255 * 3).astype(np.uint8)
    single_ori_label = np.concatenate([single_ori_label.transpose(2, 0, 1), np.expand_dims(vehicle_label, 0)], 0)
    return single_ori_label, ego_label


def pre_process(train_arr):
    img_tensor = transforms.ToTensor()(train_arr[:3, :, :].astype(np.uint8).transpose(1, 2, 0))
    # img_tensor = transforms.Normalize((0.096, 0.096, 0.096), (0.310, 0.310, 0.310))(img_tensor)
    # possibility_tensor = transforms.Normalize((0.096, 0.063, 0.050), (0.249, 0.160, 0.122))(torch.from_numpy(train_arr[3:, :, :]))
    possibility_tensor = torch.from_numpy(train_arr[3:, :, :])
    pro_train_arr = torch.cat([img_tensor, possibility_tensor], 0)
    pro_train_arr = pro_train_arr.numpy()
    return pro_train_arr


if __name__ == '__main__':
    conn, cursor = init_DB("Interaction_Intersection_EP0_Scenario_DB")
    area_range = 144
    single_range = 36
    ratio = 1.0
    alp_drive = 10
    bet_drive = 4
    alp_others = 10
    bet_others = 4

    drive_mask = np.ones((int(area_range / ratio), int(area_range / ratio), 3), np.uint8) * 255
    road_mask = np.ones((int(area_range / ratio), int(area_range / ratio), 3), np.uint8) * 255
    center_x, center_y = 1003, 995

    draw_drivable_area(cursor, drive_mask, center_x, center_y, int(area_range / ratio), ratio)
    draw_map_part(cursor, road_mask, [center_x - area_range / 2, center_x + area_range / 2],
                  [center_y - area_range / 2, center_y + area_range / 2], center_x, center_y, ratio)

    drive_label = (np.sum(drive_mask, 2) != 255 * 3).astype(np.uint8)
    road_label = (np.sum(road_mask, 2) != 255 * 3).astype(np.uint8)

    for k in range(0, 8):
        table = "_" + str(k)
        print(table)
        stampsql = "select time_stamp from Traffic_timing_state" + table
        cursor.execute(stampsql)
        timestampresult = cursor.fetchall()
        timestamp_set = set()
        for i in range(len(timestampresult)):
            timestamp_set.add(timestampresult[i][0])
        timestamp_list = list(timestamp_set)
        timestamp_list.sort()
        print(len(timestamp_list))
        max_len = 0

        for index in range(69, len(timestamp_list) - 10, 10):
            print("index: ", index)
            flag = 0
            timestamp_now = timestamp_list[index]
            timestamp_past = timestamp_list[index - 60]
            veh_collect = []
            for t in range(7):
                veh_collect.append(list(utils.SearchTrafficParticipantByTime(cursor,
                                                                             timestamp_list[index - 60 + t * 5],
                                                                             table)))
            vehicle_fu_list_1 = utils.SearchTrafficParticipantByTime(cursor, timestamp_list[index - 20], table)
            vehicle_fu_list_2 = utils.SearchTrafficParticipantByTime(cursor, timestamp_list[index - 10], table)
            vehicle_fu_list_3 = utils.SearchTrafficParticipantByTime(cursor, timestamp_now, table)
            vehicle_fu_list = list(set(vehicle_fu_list_1) & set(vehicle_fu_list_2) & set(vehicle_fu_list_3))
            for v in veh_collect:
                vehicle_fu_list = list(set(vehicle_fu_list) & set(v))
            if len(vehicle_fu_list) < 4 or len(vehicle_fu_list) > 16:
                continue

            vehicle_mask_1 = np.ones((int(area_range / ratio), int(area_range / ratio), 3), np.uint8) * 255
            vehicle_mask_1 = draw_vehicle_and_trajectory(cursor, vehicle_mask_1, center_x, center_y,
                                                         timestamp_list[index - 20], table, area_range,
                                                         ratio, vehicle_fu_list)
            vehicle_label_1 = (np.sum(vehicle_mask_1, 2) != 255 * 3).astype(np.uint8)


            vehicle_mask_2 = np.ones((int(area_range / ratio), int(area_range / ratio), 3), np.uint8) * 255
            vehicle_mask_2 = draw_vehicle_and_trajectory(cursor, vehicle_mask_2, center_x, center_y,
                                                         timestamp_list[index - 10], table, area_range,
                                                         ratio, vehicle_fu_list)
            vehicle_label_2 = (np.sum(vehicle_mask_2, 2) != 255 * 3).astype(np.uint8)


            vehicle_mask_3 = np.ones((int(area_range / ratio), int(area_range / ratio), 3), np.uint8) * 255
            vehicle_mask_3 = draw_vehicle_and_trajectory(cursor, vehicle_mask_3, center_x, center_y,
                                                         timestamp_now, table, area_range,
                                                         ratio, vehicle_fu_list)
            vehicle_label_3 = (np.sum(vehicle_mask_3, 2) != 255 * 3).astype(np.uint8)


            mask_ori_label = np.concatenate([np.expand_dims(drive_label, 0),
                                             np.expand_dims(road_label, 0),
                                             np.expand_dims(vehicle_label_1, 0),
                                             np.expand_dims(vehicle_label_2, 0),
                                             np.expand_dims(vehicle_label_3, 0)], axis=0)

            label_fig_1 = np.ones((int(area_range / ratio), int(area_range / ratio), 3), np.uint8) * 255
            label_fig_2 = np.ones((int(area_range / ratio), int(area_range / ratio), 3), np.uint8) * 255
            label_fig_3 = np.ones((int(area_range / ratio), int(area_range / ratio), 3), np.uint8) * 255

            label_fig_1[mask_ori_label[0] == 1] = [220, 220, 220]
            label_fig_1[mask_ori_label[1] == 1] = [192, 192, 192]
            label_fig_1[mask_ori_label[2] == 1] = [65, 105, 225]

            label_fig_2[mask_ori_label[0] == 1] = [220, 220, 220]
            label_fig_2[mask_ori_label[1] == 1] = [192, 192, 192]
            label_fig_2[mask_ori_label[3] == 1] = [65, 105, 225]

            label_fig_3[mask_ori_label[0] == 1] = [220, 220, 220]
            label_fig_3[mask_ori_label[1] == 1] = [192, 192, 192]
            label_fig_3[mask_ori_label[4] == 1] = [65, 105, 225]

            npy_pre = np.zeros((7, 16, 6, int(single_range / ratio), int(single_range / ratio)), np.float)
            mask_pre = np.zeros((7, 16), np.uint8)
            if not os.path.exists("./BEVImgData_01_4_100/" + str(k) + "/" + str(timestamp_now)):
                os.makedirs("./BEVImgData_01_4_100/" + str(k) + "/" + str(timestamp_now))
            if not os.path.exists("./BEVData_01_4_100"):
                os.makedirs("./BEVData_01_4_100/")
            if not os.path.exists("./BEVImgLabel_01_4_100/" + str(k) + "/" + str(timestamp_now)):
                os.makedirs("./BEVImgLabel_01_4_100/" + str(k) + "/" + str(timestamp_now))
            if not os.path.exists("./BEVMaskData_01_4_100"):
                os.makedirs("./BEVMaskData_01_4_100")
            if not os.path.exists("./BEVLabel_01_4_100"):
                os.makedirs("./BEVLabel_01_4_100")
            for time_idx in range(7):
                for veh_idx in range(len(vehicle_fu_list)):
                    timestamp = timestamp_past + 500 * time_idx
                    veh = vehicle_fu_list[veh_idx]
                    print(timestamp, veh)
                    local_x, local_y, orien = utils.SearchTrafficParticipantTiming(cursor, veh, timestamp, table)
                    single_ori_label, ego_label = ge_single_label_from_whole(cursor, veh, timestamp,
                                                                             table, area_range, single_range,
                                                                             mask_ori_label[:2, :, :], ratio)
                    p_ref_drive = np.random.beta(alp_drive, bet_drive,
                                                 (3, int(single_range / ratio), int(single_range / ratio)))
                    # p_ref_others = np.random.beta(alp_others, bet_others, (3, int(area_range / ratio), int(area_range / ratio)))
                    mask_ge_conf = np.zeros((3, int(single_range / ratio), int(single_range / ratio)), np.float)
                    # mask_ge_conf[mask_ori_label == 1] = p_ref_drive[mask_ori_label == 1]
                    # mask_ge_conf[mask_ori_label == 0] = 1 - p_ref_drive[mask_ori_label == 0]
                    mask_ge_conf[0][single_ori_label[0] == 1] = p_ref_drive[0][single_ori_label[0] == 1]
                    mask_ge_conf[0][single_ori_label[0] == 0] = 1 - p_ref_drive[0][single_ori_label[0] == 0]
                    mask_ge_conf[1:][np.logical_and(mask_ge_conf[0] > 0.5, single_ori_label[1:] == 1)] = \
                        p_ref_drive[1:][np.logical_and(mask_ge_conf[0] > 0.5, single_ori_label[1:] == 1)]
                    mask_ge_conf[1:][np.logical_and(mask_ge_conf[0] > 0.5, single_ori_label[1:] == 0)] = \
                        1 - p_ref_drive[1:][np.logical_and(mask_ge_conf[0] > 0.5, single_ori_label[1:] == 0)]

                    mask_ge_conf = mask_ge_conf.transpose(1, 2, 0)
                    mask_ge_conf = rotate_bev_label(mask_ge_conf, orien)
                    ego_label = rotate_bev_label(ego_label, orien)
                    mask_ge_label = (mask_ge_conf > 0.5).astype(np.uint8)
                    mask_ge_label = mask_ge_label.transpose(2, 0, 1)

                    mask_fig = np.ones((int(single_range / ratio), int(single_range / ratio), 3), np.uint8) * 255
                    mask_fig[mask_ge_label[0] == 1] = [220, 220, 220]
                    mask_fig[mask_ge_label[1] == 1] = [192, 192, 192]
                    mask_fig[mask_ge_label[2] == 1] = [65, 105, 225]
                    mask_fig[ego_label == 1] = [65, 179, 113]

                    data_ori = np.concatenate([mask_fig.astype(np.uint8).transpose(2, 0, 1),
                                               mask_ge_conf.transpose(2, 0, 1)], axis=0)
                    npy_pre[time_idx][veh_idx] = pre_process(data_ori)
                    mask_pre[time_idx][veh_idx] = 1
                    cv2.imwrite("./BEVImgData_01_4_100/" + str(k) + "/" + str(timestamp_now) + "/" +
                                str(timestamp) + "_" + str(veh) + ".png", mask_fig[:, :, ::-1])
                    # cv2.imshow("display", label_fig[:, :, ::-1])
                    # cv2.waitKey(0)

            cv2.imwrite("./BEVImgLabel_01_4_100/" + str(k) + "/" + str(timestamp_now) + "/" +
                        str(timestamp_list[index - 20]) + ".png", label_fig_1[:, :, ::-1])
            cv2.imwrite("./BEVImgLabel_01_4_100/" + str(k) + "/" + str(timestamp_now) + "/" +
                        str(timestamp_list[index - 10]) + ".png", label_fig_2[:, :, ::-1])
            cv2.imwrite("./BEVImgLabel_01_4_100/" + str(k) + "/" + str(timestamp_now) + "/" +
                        str(timestamp_now) + ".png", label_fig_3[:, :, ::-1])
            np.save("./BEVData_01_4_100/" + str(k) + "_" + str(timestamp_now) + ".npy", npy_pre)
            np.save("./BEVMaskData_01_4_100/" + str(k) + "_" + str(timestamp_now) + ".npy", mask_pre)
            np.save("./BEVLabel_01_4_100/" + str(k) + "_" + str(timestamp_now) + ".npy", mask_ori_label)

    # mask_ori_label = np.concatenate([np.expand_dims(drive_label, 0),
    #                                  np.expand_dims(road_label, 0)], axis=0)
    # label_fig_1 = np.ones((int(area_range / ratio), int(area_range / ratio), 3), np.uint8) * 255
    # label_fig_1[mask_ori_label[0] == 1] = [220, 220, 220]
    # label_fig_1[mask_ori_label[1] == 1] = [192, 192, 192]
    # cv2.imwrite("label_map.png", label_fig_1[:, :, ::-1])