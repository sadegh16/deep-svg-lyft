from collections import defaultdict
from typing import List, Optional, Tuple
import copy
import cv2
import numpy as np
import warnings
from ..data.filter import filter_tl_faces_by_status
from ..data.map_api import MapAPI
from ..geometry import rotation33_as_yaw, transform_point,linear_path_to_tensor, transform_points, world_to_image_pixels_matrix,yaw_as_rotation33,crop_tensor
from .rasterizer import Rasterizer
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import torch

# sub-pixel drawing precision constants
CV2_SHIFT = 8  # how many bits to shift in drawing
CV2_SHIFT_VALUE = 2 ** CV2_SHIFT
MAX_LANE_SIZE=200
MAX_LANE_NUM=20
def zero_point_extend(unsized_lane):
    if len(unsized_lane)>MAX_LANE_SIZE:
        valid_size=MAX_LANE_SIZE
    else: valid_size=len(unsized_lane)
    sized_lane=np.zeros((MAX_LANE_SIZE,2))
    sized_lane[:valid_size]=unsized_lane[:valid_size]
    return sized_lane,valid_size


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
#     print(vector)
    np.seterr(divide='ignore',invalid='ignore')
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2,inall=False):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
#     print(inall)
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_ego_as_agent(frame: np.ndarray) -> np.ndarray:  # TODO this can be useful to have around
    """
    Get a valid agent with information from the frame AV. Ford Fusion extent is used

    Args:
        frame (np.ndarray): the frame we're interested in

    Returns: an agent np.ndarray of the AV

    """
    ego_agent = np.zeros(1, dtype=AGENT_DTYPE)
    ego_agent[0]["centroid"] = frame["ego_translation"][:2]
    ego_agent[0]["yaw"] = rotation33_as_yaw(frame["ego_rotation"])
    ego_agent[0]["extent"] = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
    return ego_agent


def draw_boxes(
    raster_size: Tuple[int, int],
    world_to_image_space: np.ndarray,
    agents: np.ndarray,
    im
) -> np.ndarray:
    """
    Draw multiple boxes in one sweep over the image.
    Boxes corners are extracted from agents, and the coordinates are projected in the image plane.
    Finally, cv2 draws the boxes.

    Args:
        raster_size (Tuple[int, int]): Desired output image size
        world_to_image_space (np.ndarray): 3x3 matrix to convert from world to image coordinated
        agents (np.ndarray): array of agents to be drawn
        color (Union[int, Tuple[int, int, int]]): single int or RGB color

    Returns:
        np.ndarray: the image with agents rendered. RGB if color RGB, otherwise GRAY
    """

#     print(len(agents))
    box_world_coords = np.zeros((len(agents), 4, 2))
    corners_base_coords = np.asarray([[-1, -1], [-1, 1], [1, 1], [1, -1]])

    # compute the corner in world-space (start in origin, rotate and then translate)
    for idx, agent in enumerate(agents):
        corners = corners_base_coords * agent["extent"][:2] / 2  # corners in zero
        r_m = yaw_as_rotation33(agent["yaw"])
        box_world_coords[idx] = transform_points(corners, r_m) + agent["centroid"][:2]

    box_image_coords = transform_points(box_world_coords.reshape((-1, 2)), world_to_image_space)

    # fillPoly wants polys in a sequence with points inside as (x,y)
    box_image_coords = box_image_coords.reshape((-1, 4, 2)).astype(np.int64)
    cv2.fillPoly(im, box_image_coords, color=255)
#     return im

def elements_within_bounds(center: np.ndarray, bounds: np.ndarray, half_extent: float) -> np.ndarray:
    """
    Get indices of elements for which the bounding box described by bounds intersects the one defined around
    center (square with side 2*half_side)

    Args:
        center (float): XY of the center
        bounds (np.ndarray): array of shape Nx2x2 [[x_min,y_min],[x_max, y_max]]
        half_extent (float): half the side of the bounding box centered around center

    Returns:
        np.ndarray: indices of elements inside radius from center
    """
    x_center, y_center = center

    x_min_in = x_center > bounds[:, 0, 0] - half_extent
    y_min_in = y_center > bounds[:, 0, 1] - half_extent
    x_max_in = x_center < bounds[:, 1, 0] + half_extent
    y_max_in = y_center < bounds[:, 1, 1] + half_extent
    return np.nonzero(x_min_in & y_min_in & x_max_in & y_max_in)[0]


def cv2_subpixel(coords: np.ndarray) -> np.ndarray:
    """
    Cast coordinates to numpy.int but keep fractional part by previously multiplying by 2**CV2_SHIFT
    cv2 calls will use shift to restore original values with higher precision

    Args:
        coords (np.ndarray): XY coords as float

    Returns:
        np.ndarray: XY coords as int for cv2 shift draw
    """
    coords = coords * CV2_SHIFT_VALUE
    coords = coords.astype(np.int)
    return coords



class MySemanticRasterizer(Rasterizer):
    """
    Rasteriser for the vectorised semantic map (generally loaded from json files).
    """

    def __init__(
        self,
        raster_size: Tuple[int, int],
        pixel_size: np.ndarray,
        ego_center: np.ndarray,
        semantic_map_path: str,
        world_to_ecef: np.ndarray,
    ):
        self.raster_size = raster_size
        self.pixel_size = pixel_size
        self.ego_center = ego_center

        self.world_to_ecef = world_to_ecef

        self.proto_API = MapAPI(semantic_map_path, world_to_ecef)

        self.bounds_info = self.get_bounds()

    # TODO is this the right place for this function?
    def get_bounds(self) -> dict:
        """
        For each elements of interest returns bounds [[min_x, min_y],[max_x, max_y]] and proto ids
        Coords are computed by the MapAPI and, as such, are in the world ref system.

        Returns:
            dict: keys are classes of elements, values are dict with `bounds` and `ids` keys
        """
        lanes_ids = []
        crosswalks_ids = []

        lanes_bounds = np.empty((0, 2, 2), dtype=np.float)  # [(X_MIN, Y_MIN), (X_MAX, Y_MAX)]
        crosswalks_bounds = np.empty((0, 2, 2), dtype=np.float)  # [(X_MIN, Y_MIN), (X_MAX, Y_MAX)]

        for element in self.proto_API:
            element_id = MapAPI.id_as_str(element.id)

            if self.proto_API.is_lane(element):
                lane = self.proto_API.get_lane_coords(element_id)
                x_min = min(np.min(lane["xyz_left"][:, 0]), np.min(lane["xyz_right"][:, 0]))
                y_min = min(np.min(lane["xyz_left"][:, 1]), np.min(lane["xyz_right"][:, 1]))
                x_max = max(np.max(lane["xyz_left"][:, 0]), np.max(lane["xyz_right"][:, 0]))
                y_max = max(np.max(lane["xyz_left"][:, 1]), np.max(lane["xyz_right"][:, 1]))

                lanes_bounds = np.append(lanes_bounds, np.asarray([[[x_min, y_min], [x_max, y_max]]]), axis=0)
                lanes_ids.append(element_id)

            if self.proto_API.is_crosswalk(element):
                crosswalk = self.proto_API.get_crosswalk_coords(element_id)
                x_min = np.min(crosswalk["xyz"][:, 0])
                y_min = np.min(crosswalk["xyz"][:, 1])
                x_max = np.max(crosswalk["xyz"][:, 0])
                y_max = np.max(crosswalk["xyz"][:, 1])

                crosswalks_bounds = np.append(
                    crosswalks_bounds, np.asarray([[[x_min, y_min], [x_max, y_max]]]), axis=0,
                )
                crosswalks_ids.append(element_id)

        return {
            "lanes": {"bounds": lanes_bounds, "ids": lanes_ids},
            "crosswalks": {"bounds": crosswalks_bounds, "ids": crosswalks_ids},
        }
    
 
    def check_if_target_is_in_this_lane(self,left_func,right_func,boundry,ego_translation):
        observation=ego_translation
        if  not ((left_func(0))*(right_func(0))<=1 and \
                        boundry[0][0]-1<=observation[0]<=boundry[1][0]+1 and \
                        boundry[0][1]-1<=observation[1]<=boundry[1][1]+1 ):

            return False
        return True


    def draw_recur_lanes(self,glob_id,lanes_lines,world_to_image_space,img,active_tl_ids,drwaed_lanes,
                         left_lanes,right_lanes,left_path=[],right_path=[],depth=0,):
#         if drwaed_lanes.get(str(glob_id)): return
#         if str(glob_id) not in self.normalized_left_borders: return

        drwaed_lanes[str(glob_id)]=True
        lane = self.proto_API[glob_id].element.lane
        # get image coords
        lane_coords = self.proto_API.get_lane_coords(glob_id)
#         xy_left = cv2_subpixel(transform_points(lane_coords["xyz_left"][:, :2], world_to_image_space))
#         xy_right = cv2_subpixel(transform_points(lane_coords["xyz_right"][:, :2], world_to_image_space))
#         lanes_area = np.vstack((xy_left, np.flip(xy_right, 0)))  # start->end left then end->start right

    
#         # Note(lberg): this called on all polygons skips some of them, don't know why
#         cv2.fillPoly(img, [lanes_area], (17, 17, 31), lineType=cv2.LINE_AA, shift=CV2_SHIFT)

#         lane_type = "default"  # no traffic light face is controlling this lane
#         lane_tl_ids = set([MapAPI.id_as_str(la_tc) for la_tc in lane.traffic_controls])
#         for tl_id in lane_tl_ids.intersection(active_tl_ids):
#             if self.proto_API.is_traffic_face_colour(tl_id, "red"):
#                 lane_type = "red"
#             elif self.proto_API.is_traffic_face_colour(tl_id, "green"):
#                 lane_type = "green"
#             elif self.proto_API.is_traffic_face_colour(tl_id, "yellow"):
#                 lane_type = "yellow"
#         lanes_lines[lane_type].extend([xy_left, xy_right])
        
        
        if str(glob_id) not in self.normalized_left_borders:
            lane_coords = self.proto_API.get_lane_coords(glob_id)
            self.normalized_left_borders[str(glob_id)]=self.normalize_border(lane_coords["xyz_left"][:, :2],
                                                                             self.ego_translation,self.ego_yaw,world_to_image_space)
            self.normalized_right_borders[str(glob_id)]=self.normalize_border(lane_coords["xyz_right"][:, :2],
                                                                              self.ego_translation,self.ego_yaw,world_to_image_space)
            
        xy_left=self.normalized_left_borders[str(glob_id)]
        xy_right=self.normalized_right_borders[str(glob_id)]
        xy_left=crop_tensor(xy_left, (224,224))
        xy_right=crop_tensor(xy_right, (224,224))
        
        #         b = copy.deepcopy(a)
        
        tmp_left_path=left_path+xy_left.tolist()
        tmp_right_path=right_path+xy_right.tolist()
            
        added=False
        
        if depth<6: 
            for i in range(len(lane.lanes_ahead)):
#                 if str(lane.lanes_ahead[i].id) in self.normalized_left_borders:
                added=True
                self.draw_recur_lanes(lane.lanes_ahead[i].id,lanes_lines,world_to_image_space,img,active_tl_ids,drwaed_lanes,
                                      left_lanes,right_lanes,tmp_left_path,tmp_right_path,depth=depth+1)
        
        if depth<=0:
            if lane.adjacent_lane_change_right.id!=b"" :
                self.draw_recur_lanes(lane.adjacent_lane_change_right.id,lanes_lines,world_to_image_space,img,active_tl_ids,
                                      drwaed_lanes,left_lanes,right_lanes,depth=depth+1)
            if lane.adjacent_lane_change_left.id!=b"" :
                self.draw_recur_lanes(lane.adjacent_lane_change_left.id,lanes_lines,world_to_image_space,img,active_tl_ids,
                                      drwaed_lanes,left_lanes,right_lanes,depth=depth+1)
        
        if added==False:
            left_lanes.append(tmp_left_path)
            right_lanes.append(tmp_right_path)
            
        
        
        
    def find_lanes_of_points(self,center_world,world_to_image_space, tl_faces,ego_translation,img,raster_radius,active_tl_ids,lanes_lines,
                            left_funcs,right_funcs,lane_indicies,ego_yaw,used_tail):
        selected_indicies=[]
        for i in range(len(lane_indicies)):
                idx=lane_indicies[i]
                lane = self.proto_API[self.bounds_info["lanes"]["ids"][idx]].element.lane

                # get image coords
                lane_coords = self.proto_API.get_lane_coords(self.bounds_info["lanes"]["ids"][idx])
                left_func=left_funcs[i]
                right_func=right_funcs[i]
                
                boundry=self.bounds_info["lanes"]["bounds"][idx]
        
                if self.check_if_target_is_in_this_lane(left_func,right_func,boundry,ego_translation) :
#                 self.check_if_target_is_alligned_with_lane(lane_coords["xyz_left"][:, :2],ego_translation,ego_yaw):
        
#                     start=self.find_end_point_of_the_lane(lane_coords["xyz_left"][:, :2],ego_translation,ego_yaw)
                    used_tail[idx]=0

                             
                            
                    selected_indicies.append(idx)
                    

        return selected_indicies
    

    def normalize_border(self,border,ego_translation,ego_yaw,world_to_image_space=None):
#       
        if  world_to_image_space is None:
            return transform_points(border-ego_translation[:2], yaw_as_rotation33(-ego_yaw))
        return transform_points(border, world_to_image_space)
        
    
    def rasterize(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tl_faces: List[np.ndarray],
        agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # TODO TR_FACES
            
        if agent is None:
            ego_translation = history_frames[0]["ego_translation"]
            ego_yaw = rotation33_as_yaw(history_frames[0]["ego_rotation"])
        else:
            ego_translation = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            ego_yaw = agent["yaw"]

        world_to_image_space = world_to_image_pixels_matrix(
            self.raster_size, self.pixel_size, ego_translation, ego_yaw, self.ego_center,
        )

        # get XY of center pixel in world coordinates
        center_pixel = np.asarray(self.raster_size) * (0.5, 0.5)
        center_world = transform_point(center_pixel, np.linalg.inv(world_to_image_space))


        
        sem_im = self.render_semantic_map(center_world, world_to_image_space, history_tl_faces[0],ego_translation,ego_yaw,)
        return sem_im.astype(np.float32) / 255

    
    
    def render_semantic_map(
        self, center_world: np.ndarray, world_to_image_space: np.ndarray, tl_faces: np.ndarray,ego_translation,ego_yaw,
    ) -> np.ndarray:
        """Renders the semantic map at given x,y coordinates.

        Args:
            center_world (np.ndarray): XY of the image center in world ref system
            world_to_image_space (np.ndarray):
        Returns:
            np.ndarray: RGB raster

        """

        
        img = 255 * np.ones(shape=(self.raster_size[1], self.raster_size[0], 3), dtype=np.uint8)
        # filter using half a radius from the center
        raster_radius = float(np.linalg.norm(self.raster_size * self.pixel_size)) / 2

        # get active traffic light faces
        active_tl_ids = set(filter_tl_faces_by_status(tl_faces, "ACTIVE")["face_id"].tolist())
        # plot lanes
        lanes_lines = defaultdict(list)
        left_funcs=[]
        right_funcs=[]   
        lane_indicies=[]  
        continued_lane=[]
        used_tail=dict()
        self.normalized_left_borders=dict()
        self.normalized_right_borders=dict()
        self.ego_translation=ego_translation
        self.ego_yaw=ego_yaw
        for idx in elements_within_bounds(center_world, self.bounds_info["lanes"]["bounds"], raster_radius):
                lane = self.proto_API[self.bounds_info["lanes"]["ids"][idx]].element.lane
                # get image coords
                lane_coords = self.proto_API.get_lane_coords(self.bounds_info["lanes"]["ids"][idx])
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', np.RankWarning)
                    normalized_left=self.normalize_border(lane_coords["xyz_left"][:, :2],ego_translation,ego_yaw)
                    normalized_right=self.normalize_border(lane_coords["xyz_right"][:, :2],ego_translation,ego_yaw)
                    glob_id=str(self.bounds_info["lanes"]["ids"][idx])
                    

                    
                    left_func = np.poly1d(np.polyfit(normalized_left[:, 0], normalized_left[:, 1], 3))
                    right_func = np.poly1d(np.polyfit(normalized_right[:, 0], normalized_right[:, 1], 3))
                    normalized_left=self.normalize_border(lane_coords["xyz_left"][:, :2],ego_translation,ego_yaw,world_to_image_space)
                    normalized_right=self.normalize_border(lane_coords["xyz_right"][:, :2],ego_translation,ego_yaw,world_to_image_space)
                    self.normalized_left_borders[glob_id]=normalized_left
                    self.normalized_right_borders[glob_id]=normalized_right
        
                    lane_indicies.append(idx)
                    left_funcs.append(left_func)
                    right_funcs.append(right_func)
                    
        selected_indicies=self.find_lanes_of_points(center_world,world_to_image_space,tl_faces,ego_translation,
                                                    img,raster_radius,active_tl_ids,lanes_lines,
                                 left_funcs,right_funcs,lane_indicies,ego_yaw,used_tail)
    
        drwaed_lanes=dict()
        left_lanes=[]
        right_lanes=[]
        
        for idx in selected_indicies:
            
            curr=self.bounds_info["lanes"]["ids"][idx]
            self.draw_recur_lanes(curr,lanes_lines,world_to_image_space,img,active_tl_ids,drwaed_lanes,
                                  left_lanes,right_lanes,)
        
        
#         print(len(left_lanes))
#         for i in range(12):
#             print(len(right_lanes[i]))
        
#         cv2.polylines(img, lanes_lines["default"], False, (255, 217, 82), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
#         cv2.polylines(img, lanes_lines["green"], False, (0, 255, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
#         cv2.polylines(img, lanes_lines["yellow"], False, (255, 255, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
#         cv2.polylines(img, lanes_lines["red"], False, (255, 0, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        
        
        final_lane=np.zeros((MAX_LANE_NUM,MAX_LANE_SIZE,2))
        final_lane_size=np.zeros((MAX_LANE_NUM,1))
        final_lane_num=0
        borders=[]
        for i in range(len(left_lanes)):
            if final_lane_num>=MAX_LANE_NUM:break
            if len(left_lanes[i])!=0: 
                borders.append(left_lanes[i])
                final_lane[final_lane_num],final_lane_size[final_lane_num]=zero_point_extend(left_lanes[i])
                final_lane_num+=1
            if final_lane_num>=MAX_LANE_NUM:break
            if len(right_lanes[i])!=0:
                borders.append(right_lanes[i])
                final_lane[final_lane_num],final_lane_size[final_lane_num]=zero_point_extend(right_lanes[i])
                final_lane_num+=1
                
#         if self.use_scene:
#         path_type=[0]*len(cnt_lines_norm)
#         print(np.array(left_lanes[0]))
#         print(np.array(left_lanes[0]).shape)
        if len(borders)==0:
            borders.append([[0,0],[0,12],[0,24]])
            
        path=torch.cat([linear_path_to_tensor(np.array(lane), -1)  for lane in borders], 0)
        
#         print(path[0])
#         print(path[1].shape)
#         print(path[2].shape)
        
        
        self.final_lane,self.final_lane_size,self.final_lane_num=final_lane,final_lane_size,final_lane_num
        self.path=path
        
        return img

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        return (in_im * 255).astype(np.uint8)