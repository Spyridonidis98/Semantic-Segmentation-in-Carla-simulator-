import glob
from math import sin
import os
import sys
import numpy as np
import cv2 as cv
from time import sleep
import random
import pygame 
from matplotlib import pyplot as plt 
import tensorflow as tf 
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0],True)
tf.keras.mixed_precision.set_global_policy('mixed_float16')
from focal_loss import SparseCategoricalFocalLoss
from dataloader import labels2RGB
import queue

try:
    sys.path.append(glob.glob('../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
#global params
# ==============================================================================
# -- ClientSideWaypoints ---------------------------------------------------
# ==============================================================================
class ClientSideWaypoints(object):  
    """This is a module responsible for drawing waypoints from a camera perspective on pygame surface.
    """
    BB_COLOR = (27, 64, 94)
    VIEW_WIDTH = None
    VIEW_HEIGHT = None

    @classmethod 
    def get_waypoints(cls, waypoints, camera):
        """
        get waypoints uv cordinates from world cordinates
        """
        cls.VIEW_WIDTH = int(camera.attributes["image_size_x"])
        cls.VIEW_HEIGHT = int(camera.attributes["image_size_y"])
        waypoints_uv = [ClientSideWaypoints.get_waypoint(waypoint, camera) for waypoint in waypoints]

        return waypoints_uv
    
    @staticmethod
    def get_waypoint(waypoint, camera):
        loc = waypoint.transform.location
        cords = np.transpose(np.matrix([loc.x, loc.y, loc.z, 1]))

        cords_x_y_z = ClientSideBoundingBoxes._world_to_sensor(cords, camera)
        #cordinates relative to the camera 
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @classmethod
    def draw_waypoints(cls, display, waypoints_uv, render_pos):
        wp_surface = pygame.Surface((cls.VIEW_WIDTH, cls.VIEW_HEIGHT))
        wp_surface.set_colorkey((0, 0, 0))
        for wp in waypoints_uv:
            c  = (wp[0, 0], wp[0, 1])
            pygame.draw.circle(wp_surface, cls.BB_COLOR, c, 5)

        display.blit(wp_surface, render_pos)

# ==============================================================================
# -- ClientSideBoundingBoxes ---------------------------------------------------
# ==============================================================================
class ClientSideBoundingBoxes(object):    
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """
    BB_COLOR = (248, 64, 24)
    VIEW_WIDTH = None
    VIEW_HEIGHT = None

    
    @classmethod
    def get_2d_bounding_box(cls, segmented_image):
        """
        Returns the 2d bounding box of the segmented image 
        """
        
        width = segmented_image.shape[1]
        height = segmented_image.shape[0]  
        cls.VIEW_WIDTH = width
        cls.VIEW_HEIGHT = height
        width_left = 0
        width_right = 0 
        height_up = 0 
        height_down = 0

        for i in range(width):
            col = segmented_image[:, i, :]
            for i2 in range(len(col)):
                if ( col[i2] == np.array([142, 0, 0]) ).all():
                    width_left = i
                    break
            else:
                continue
            break

        for i in reversed(range(width)):
            col = segmented_image[:, i, :]
            for i2 in range(len(col)):
                if ( col[i2] == np.array([142, 0, 0]) ).all():
                    width_right = i
                    break
            else:
                continue
            break

        center_x = (width_left+width_right)//2
        width = width_right - width_left

        for i in range(height):
            row  = segmented_image[i, :, :]
            for i2 in range(len(row)):
                if ( row[i2] == np.array([142, 0, 0]) ).all():
                    height_up = i
                    break
            else:
                continue
            break

        for i in reversed(range(height)):
            row = segmented_image[i, :, :]
            for i2 in range(len(row)):
                if ( row[i2] == np.array([142, 0, 0]) ).all():
                    height_down = i
                    break
            else:
                continue
            break

        center_y  = (height_up+height_down)//2
        height = height_down - height_up
        return (center_x, center_y, width, height)


    @classmethod
    def draw_2d_bounding_box(cls, display, bounding_box, render_pos):
        """
        Draws 2d bounding box, with center x, y, width, height 
        """
        
        bb_surface = pygame.Surface((cls.VIEW_WIDTH, cls.VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))

        points = []
        center_x = bounding_box[0]
        center_y = bounding_box[1]
        box_width = bounding_box[2]
        box_height = bounding_box[3]

        points.append((center_x-box_width/2, center_y-box_height/2))
        points.append((center_x+box_width/2, center_y-box_height/2))
        points.append((center_x+box_width/2, center_y+box_height/2))
        points.append((center_x-box_width/2, center_y+box_height/2))

        pygame.draw.line(bb_surface, cls.BB_COLOR, points[0], points[1])
        pygame.draw.line(bb_surface, cls.BB_COLOR, points[1], points[2])
        pygame.draw.line(bb_surface, cls.BB_COLOR, points[2], points[3])
        pygame.draw.line(bb_surface, cls.BB_COLOR, points[3], points[0])


        display.blit(bb_surface, render_pos)


    @classmethod
    def get_3d_bounding_boxes(cls, vehicles, camera):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """
        cls.VIEW_WIDTH = int(camera.attributes["image_size_x"])
        cls.VIEW_HEIGHT = int(camera.attributes["image_size_y"])

        bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box(vehicle, camera) for vehicle in vehicles]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        bounding_boxes = [bb for bb in bounding_boxes if all(0<=bb[:, 0]) and all(bb[:, 0]<cls.VIEW_WIDTH) ]
        bounding_boxes = [bb for bb in bounding_boxes if all(0<=bb[:, 1]) and all(bb[:, 1]<cls.VIEW_HEIGHT) ]
        return bounding_boxes

    @classmethod
    def draw_3d_bounding_boxes(cls, display, bounding_boxes, render_pos):
        """
        Draws bounding boxes on pygame display.
        """

        bb_surface = pygame.Surface((cls.VIEW_WIDTH, cls.VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # draw lines
            # base
            pygame.draw.line(bb_surface, cls.BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, cls.BB_COLOR, points[1], points[2])
            pygame.draw.line(bb_surface, cls.BB_COLOR, points[2], points[3])
            pygame.draw.line(bb_surface, cls.BB_COLOR, points[3], points[0])
            # top
            pygame.draw.line(bb_surface, cls.BB_COLOR, points[4], points[5])
            pygame.draw.line(bb_surface, cls.BB_COLOR, points[5], points[6])
            pygame.draw.line(bb_surface, cls.BB_COLOR, points[6], points[7])
            pygame.draw.line(bb_surface, cls.BB_COLOR, points[7], points[4])
            # base-top
            pygame.draw.line(bb_surface, cls.BB_COLOR, points[0], points[4])
            pygame.draw.line(bb_surface, cls.BB_COLOR, points[1], points[5])
            pygame.draw.line(bb_surface, cls.BB_COLOR, points[2], points[6])
            pygame.draw.line(bb_surface, cls.BB_COLOR, points[3], points[7])

        display.blit(bb_surface, render_pos)

    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        #cordinates relative to the camera 
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """
        
        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

    @staticmethod
    def bb_to_camera_cords(vehicle, camera):
        """
        Returns vehicles bounding box conrdinates relative to the camera 
        also known as camera space 
        """
        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        return cords_y_minus_z_x

# ==============================================================================
# -- ClientSideSensors ---------------------------------------------------
# ==============================================================================
class Sensors:
    class Camera(object):
        camera_id = 0
        def __init__(self, world, vehicle , spawn_point, reder_pos = (0,0), render_dimensions = (640, 360), attach_to = None, type = 0, model=None):
            #params
            self.type = type
            self.width = render_dimensions[0]
            self.height = render_dimensions[1]
            fov = 90
            self.vehicle = vehicle
            self.model = model
            
            #camera data
            self.camera_id = self.__class__.camera_id
            self.__class__.camera_id +=1
            self.image = None #3d matrix of the data of the camera 
            self.image_frame = 0
            self.surface = None 
            self.queue = queue.Queue()
            self.render_pos = reder_pos
            self.world = world
            self.sensor_list = [
                ["sensor.camera.rgb", cc.Raw],
                ["sensor.camera.semantic_segmentation", cc.CityScapesPalette]
            ]
            self.blueprint_library = self.world.get_blueprint_library()
            cam_bp = self.blueprint_library.find(self.sensor_list[self.type][0])
            cam_bp.set_attribute("image_size_x", f"{self.width}")
            cam_bp.set_attribute("image_size_y", f"{self.height}")
            cam_bp.set_attribute("fov", str(fov))
            self.sensor = self.world.spawn_actor(cam_bp, spawn_point, attach_to = attach_to)
            #self.sensor = self.world.spawn_actor(cam_bp, spawn_point, attach_to = vehicle)
            
            #calibration
            calibration = np.identity(3)
            calibration[0, 2] = self.width / 2.0
            calibration[1, 2] = self.height / 2.0
            calibration[0, 0] = calibration[1, 1] = self.width / (2.0 * np.tan(fov * np.pi / 360.0))
            self.sensor.calibration = calibration
            
            self.set_sensor()
         
        def set_sensor(self):
            #starts listening to sensor
            self.sensor.listen(lambda image: self.parse_image(image))

        def set_camera_pos(self, location, rotation = carla.Rotation(0,0,0)):
            box = self.vehicle.bounding_box
            location = location#carla.Location(box.location) + location
            spawn_point = carla.Transform(location, rotation)
            self.sensor.set_transform(spawn_point)


        def parse_image(self, image):
                array = np.array(image.raw_data)
                array = array.reshape(self.height, self.width, 4)
                array = array[:, :, :3]
                self.image = array
                array = array[:, :, ::-1]#bgr to rgb image
                if self.type == 1:
                    image.convert(self.sensor_list[self.type][1])
                    array = np.array(image.raw_data)
                    array = array.reshape(self.height, self.width, 4)
                    array = array[:, :, :3]
                    array = array[:, :, ::-1]#bgr to rgb image
                self.queue.put(array)

        #render current sensor data to display
        def render(self, display):
            # if self.surface is not None:
            #     display.blit(self.surface, self.render_pos)
            array = self.queue.get()
            if self.model != None:
                Tags2color = {0:(0,0,142), 1:(128, 64, 128), 2:(0,0,0)}
                semantic_prediction = self.model(np.expand_dims(array, 0)).numpy()
                semantic_prediction = np.expand_dims(np.argmax(semantic_prediction, axis=3), axis=3)
                semantic_prediction = labels2RGB(semantic_prediction, Tags2color=Tags2color)[0]
                array = ( array * 0.5 + semantic_prediction * 0.5 ).astype(np.uint8)
            self.surface= pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(self.surface, self.render_pos)

        def save_image(self):
            if self.type == 0:
                cv.imwrite(f"x/{world.world.get_map().name[11:]}_{self.image_frame}.png", self.image)
            if self.type == 1:
                 cv.imwrite(f"y/{world.world.get_map().name[11:]}_{self.image_frame}.png", self.image)
            self.image_frame +=1

        def destroy(self):
            self.sensor.destroy()

# ==============================================================================
# -- ClientSideWorld ---------------------------------------------------
# ==============================================================================
class World:
    def __init__(self, world):
        self.sensor_list = []
        self.actor_list = []
        self.world = world
        
        #world settings 
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 1.0/30
        # self.settings.substepping = True
        # self.settings.max_substep_delta_time = 0.01
        #self.settings.max_substeps = 10
        self.settings.no_rendering_mode = True
        self.world.apply_settings(self.settings)

        #gets all cars blueprints 
        self.blueprint_library = self.world.get_blueprint_library()
        self.blueprints = [vehicle for vehicle in list(self.blueprint_library.filter("vehicle.*")) if vehicle.get_attribute("number_of_wheels").as_int() == 4]
        self.num_of_npc = 0

    def spawn_ego_car(self, i = 0):
        bp = self.blueprints[i]
        #bp = self.blueprint_library.filter("model3")[0]
        #spawn_point = random.choice(world.get_map().get_spawn_points())
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(bp, spawn_point)
        #self.vehicle.set_autopilot(True)
        self.actor_list.append(self.vehicle)
        #get bounding box of vehicle
        #box = self.vehicle.bounding_box
        #box.location boxes location(center of the box) relative to vehicle location 
        #box.extent boxes dimensions from the center of the box 

    def spawn_camera(self, spawn_point, render_pos = (0,0), render_dimentions = (640, 360), attach_to = None, type = 0, model = None):
        #spawn camera, z up and down, y left and right, x forward and backward 
        self.sensor_list.append(Sensors.Camera(self.world, self.vehicle, spawn_point, render_pos, render_dimentions, attach_to, type, model))

    def render(self, display):
        for sensor in self.sensor_list:
            sensor.render(display)
    
    def save_cameras_frames(self):
        for sensor in self.sensor_list:
            sensor.save_image()

    def destory_actors(self):
        print("destroying actors")
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []
        for sensor in self.sensor_list:
            sensor.destroy()
        self.sensor_list = []
        print("all clened up")

    def spawn_npc(self):
        blueprint = random.choice(self.blueprints) 
        spawn_point = self.world.get_map().get_spawn_points()[self.num_of_npc+1]#+1 because ego car spawns at 0
        npc = self.world.spawn_actor(blueprint, spawn_point)
        npc.set_autopilot(True)
        self.actor_list.append(npc)
        self.num_of_npc +=1
        # bp = self.blueprint_library.filter("Mini")[0]
        # spawn_point = self.world.get_map().get_spawn_points()[2]
        # spawn_point.location.x = 74
        # spawn_point.location.y = 7.808423
        # npc = self.world.spawn_actor(bp, spawn_point)
        # self.actor_list.append(npc)

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================
class KeyboardControl:
    def __init__(self, world):
        self.world = world
        self.control = carla.VehicleControl()
        self.reverse = False
        self.steer_cache = 0

    def parse_events(self, clock):
        self.parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())

    def parse_vehicle_keys(self, keys, milliseconds):
        if keys[pygame.K_UP]:
            if self.reverse == True:
                self.control.throttle = 0
            self.reverse = False
            self.control.throttle = min(self.control.throttle + 0.01, 1)
        

        if keys[pygame.K_DOWN]:
            #self.control.brake = min(self.control.brake + 0.2, 1)
            if self.reverse == False:
                self.control.throttle = 0
            self.reverse = True
            self.control.throttle = min(self.control.throttle + 0.01, 1)
        
        
        steer_increment = 5e-4 * milliseconds
        if keys[pygame.K_LEFT] :
            if self.steer_cache > 0:
                self.steer_cache = 0
            else:
                self.steer_cache -= steer_increment
        elif keys[pygame.K_RIGHT]:
            if self.steer_cache < 0:
                self.steer_cache = 0
            else:
                self.steer_cache += steer_increment
        else:
            self.steer_cache = 0.0
        self.steer_cache = min(0.7, max(-0.7, self.steer_cache))
        self.control.steer = round(self.steer_cache, 1)

        self.control.reverse  = self.reverse
        self.world.vehicle.apply_control(self.control)

        
# ==============================================================================
# -- VectorSpace ---------------------------------------------------
# ==============================================================================
class VectorSpace:
    #to do/ can take render dimention and render position on the screen
    BB_COLOR = (248, 64, 24)
    
    def __init__(self):
        self.ego_vehicle = VectorSpace.Vehicle(np.array([[2, 0],[2, -4],[0, -4], [0,0]]))
        

    class Vehicle:
        def __init__(self, bb_2d, ego_cars_dimentions = (2,4)):
            #1 meter equals 8 pixels
            self.bb_2d = bb_2d*8
            self.bb_2d[:, 1] = -self.bb_2d[:, 1]
            display_width = 320
            display_height = 360
            self.center_x = display_width/2-ego_cars_dimentions[0]/2
            self.center_y = display_height/2-ego_cars_dimentions[1]/2
            self.color = (255,0,0)

            
        def render(self,display, render_pos = (0,0)):
            x = render_pos[0]+self.center_x
            y = render_pos[1]+self.center_y
            pygame.draw.polygon(display, self.color, [np.add(self.bb_2d[0], [x,y]), np.add(self.bb_2d[1], [x,y]), np.add(self.bb_2d[2], [x,y]), np.add(self.bb_2d[3], [x,y])])

        
    @staticmethod
    def get_position(bb_1, bb_2, vehicle, vehicle2, camera):
        if(len(bb_1) !=0 and len(bb_2) !=0):
            bb_1 = bb_1[0]
            bb_2 = bb_2[0]
            film_cords_1 = VectorSpace.from_pixels_to_film_cords(bb_1.astype(int))
            film_cords_2 = VectorSpace.from_pixels_to_film_cords(bb_2.astype(int))

            disparity = np.concatenate(film_cords_1[:, 0] - film_cords_2[:, 0], axis=1)
            b = vehicle.bounding_box.extent.y*2
            f = 1/(2.0 * np.tan(90 * np.pi / 360.0))
            z = (f*b)/disparity
            x = np.transpose(film_cords_1[:, 0])*b/disparity
            y = np.transpose(film_cords_1[:, 1])*b*360/(disparity*640)
            world_pos = np.transpose(np.concatenate([x[0,:4],z[0,:4]]))
            return world_pos
        else:
            return None

    @staticmethod
    def from_pixels_to_film_cords(input):
        width = 640
        height = 360
        film_x = input[: ,0]/width - 0.5
        film_y = input[:, 1]/height - 0.5
        return np.concatenate([film_x, film_y], axis=1)
    
    
    def render(self, display, render_pos, actor_pos):
            self.ego_vehicle.render(display, render_pos)
            pygame.draw.line(display, self.BB_COLOR, (320+render_pos[0],0+render_pos[1]), (320+render_pos[0],360+render_pos[1]))
            if(actor_pos is not None):
                actor_pos = np.array(actor_pos)
                vilian = VectorSpace.Vehicle(actor_pos)
                vilian.color = (0,0,255)
                vilian.render(display, render_pos)
    

if __name__ == "__main__":
    #params
    width = 384*4
    height = 384*2
    
    # pygame 
    import pygame
    world = None
    
    #tensorflow 
    #load model 
    model = tf.keras.models.load_model("../../pytorch-tensorflow/tensorflow/unet_384_2", {"SparseCategoricalFocalLoss":SparseCategoricalFocalLoss(gamma=2)})

    try:
        #pygame setup 
        pygame.init()
        clock = pygame.time.Clock()
        display = pygame.display.set_mode((width,height))
        pygame.display.set_caption("game")

        #connect to server 
        client = carla.Client("localhost", 2000)
        client.set_timeout(4.0)
    
        #get world
        world = World(client.get_world())
        world.spawn_ego_car(2)
        
        #spawn cameras
        box = world.vehicle.bounding_box
        spawn_points = [carla.Transform(carla.Location(x= box.location.x - 4, y= box.location.y , z = box.location.z ), carla.Rotation(0,0,0)),
                        carla.Transform(carla.Location(x= box.location.x+box.extent.x, y= box.location.y - box.extent.y, z=box.location.z + 0.5 * box.extent.z), carla.Rotation(0, 0, 0)),
                        carla.Transform(carla.Location(x= box.location.x+box.extent.x, y= box.location.y + box.extent.y, z=box.location.z + 0.5 * box.extent.z), carla.Rotation(0, 0, 0))]
        
        render_pos = [(0,0), (0,384), (768,384)]
        
        world.spawn_camera(spawn_points[0], render_pos[0], (768, 384), attach_to=world.vehicle, type = 0, model=None)
        world.spawn_camera(spawn_points[1], render_pos[1], (768, 384), attach_to=world.vehicle, type = 0, model=model)

        for i in range(15):
            world.spawn_npc()

        world.world.tick()
        vehicles = world.world.get_actors().filter('vehicle.*')
        
        #get vehicle controler
        controller = KeyboardControl(world)

        #mains cameras potisions
        box = world.vehicle.bounding_box
        mcp = [(carla.Location(x = -10 , y= 0 , z = 2), carla.Rotation(-20,0,0)), (carla.Location(x = 0 , y= -5 , z = 2), carla.Rotation(-20,90,0)), (carla.Location(x = 5 , y= 2 , z = 2), carla.Rotation( -20,200,0)), (carla.Location(x= box.location.x+box.extent.x, y= box.location.y - box.extent.y, z=box.location.z + 0.5 * box.extent.z), carla.Rotation(0,0,0)) ]
        mcp_index = 0

        #vector space 
        vector_space = VectorSpace()
        #game loop 
        while True:
            #check events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        world.sensor_list[0].set_camera_pos(mcp[mcp_index][0], mcp[mcp_index][1])
                        mcp_index+=1
                        mcp_index = mcp_index%len(mcp)
                    if event.key == pygame.K_r:
                        if controller.reverse == False:
                            controller.reverse = True
                        else:
                            controller.reverse = False
                    if event.key == pygame.K_l:
                        world.save_cameras_frames()

            controller.parse_events(clock)
            display.fill((0, 0, 0))
            world.render(display)
            
            # bounding_boxes = ClientSideBoundingBoxes.get_3d_bounding_boxes(vehicles, world.sensor_list[0].sensor)
            # ClientSideBoundingBoxes.draw_3d_bounding_boxes(display, bounding_boxes, (0, 0))
            #bounding_boxes_cam1 = ClientSideBoundingBoxes.get_bounding_boxes(vehicles, world.camera1.sensor)
            #ClientSideBoundingBoxes.draw_bounding_boxes(display, bounding_boxes_cam1, (0, 0))
            #bounding_boxes_cam2 = ClientSideBoundingBoxes.get_bounding_boxes(vehicles, world.camera2.sensor)
            #ClientSideBoundingBoxes.draw_bounding_boxes(display, bounding_boxes_cam2, (640, 0))
            
            #actor_pos = VectorSpace.get_position(bounding_boxes_cam1, bounding_boxes_cam2, world.vehicle, world.actor_list[1], world.camera1.sensor)
            #real_actor_pos = ClientSideBoundingBoxes.bb_to_camera_cords(world.actor_list[1], world.camera1.sensor)
            # real_actor_pos = np.transpose(real_actor_pos[:,:4])
            # real_actor_pos = np.concatenate([real_actor_pos[:, 0], real_actor_pos[:, 2]], axis=1)
            # vector_space.render(display, (640, 360), actor_pos)
            # vector_space.render(display, (640+640/2, 360), real_actor_pos)
            
            # print(world.vehicle.get_location())
            # print(world.actor_list[1].get_location())
            
            pygame.display.flip()
            world.world.tick()
            #pygame.display.update()
            clock.tick(30)

    finally:
        if world is not None:
            world.destory_actors()
        print("done")

    # keys = pygame.key.get_pressed()
    # if keys[pygame.K_UP]:
    #             print("pressed up")
    #         if keys[pygame.K_DOWN]:
    #             print("pressed down")