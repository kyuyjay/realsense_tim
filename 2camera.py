import socket
import time
from math import sqrt, pi, asin, atan   
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation as R
from skspatial.objects import Plane, Points
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import cv2
import traceback
import struct
import apriltag
import argparse
from PIL import Image

HOST = '169.254.34.2'
PORT = 30002
RNG = np.random.default_rng()
NUM_CIRCLES = 4
NUM_POINTS = 500
MIN_RADIUS_STATIONARY = 23 
MAX_RADIUS_STATIONARY = 30
MIN_RADIUS_LASTMILE = 100 
MAX_RADIUS_LASTMILE = 120
MIN_RADIUS = 45
MAX_RADIUS = 60

#motion planning
into_hole = -0.032
real_sense_x = 0.08 #differences between real sense and end effector (real sense)
real_sense_y = 0.09
real_sense_z = 0.295

def connect_to_robot():
    s = socket.socket()
    s.connect((HOST,PORT))
    return s

def send_command(s,command):
    s.send(command)
    data = s.recv(1024)	
    return data

def command_builder(point, rotvec,euler, a='0.05',v='0.15'):
    print('euler')
    print(euler)
    if euler[0]<-95:
        r_x = str(point[0] + real_sense_x + 0.12 ) #to tune, +0.05 is to compensate for the tilt
    elif euler[0]>-85:
        r_x = str(point[0] + real_sense_x - 0.04 ) #to tune, +0.05 is to compensate for the tilt
    else:
        r_x = str(point[0] + real_sense_x+0.08)
    r_y = str(point[2] + real_sense_z- 0.4) #to tune
    r_z = str(-point[1] + real_sense_y ) #to tune
    phi = str(rotvec[0])
    eta = str(rotvec[1])
    zeta = str(-rotvec[2])
    command_str = 'movel(p[' + r_x +',' + r_y + ',' + r_z + ',' + phi + ',' + eta +',' + zeta + '],a=' + a + ',v=' + v + ')\n'
    print(command_str)
    command = command_str.encode()
    return command

def command_builder_back( a='0.05',v='0.2'): 
    r_x = str( 0 )  
    r_y = str(-0.3)  
    r_z = str(0)	
    command_str = 'movel(pose_trans(' + 'get_forward_kin()' + ',' + 'p[' + r_x + ',' + r_y + ',' + r_z + ',' + '0,0,0]' + '),a=' + a + ',v=' + v + ') \n'
    print(command_str)
    command = command_str.encode()
    return command

def command_builder_adjust(adjust_z, adjust_x, a='0.05',v='0.2'): 
    r_x = str( adjust_x )  
    r_y = str( 0 )  
    r_z = str( adjust_z )
    command_str = 'movel(pose_trans(' + 'get_forward_kin()' + ',' + 'p[' + r_x + ',' + r_y + ',' + r_z + ',' + '0,0,0]' + '),a=' + a + ',v=' + v + ') \n'
    print(command_str)
    command = command_str.encode()
    return command

def command_builder_rotate(point, elev, a='0.05',v='0.15'):
    elev = str(elev)
    command_str = 'movel(pose_trans(' + 'get_forward_kin()' + ',' + 'p[' + '0' + ',' + '0' + ',' + '0' + ',' + elev + ',0,0]' + '),a=' + a + ',v=' + v + ') \n'
    print(command_str)
    command = command_str.encode()

    return command

def command_builder_forward(point, elev, a='0.05',v='0.15'):
    r_x = str((-point[0] + 0.03)) 
    r_y = str((point[2] -0.03)) 
    r_z = str(point[1] -0.06)
    elev = str(elev)
    command_str = 'movel(pose_trans(' + 'get_forward_kin()' + ',' + 'p[' + r_x + ',' + r_y + ',' + r_z + ',' + '0,0,0]' + '),a=' + a + ',v=' + v + ') \n'
    print(command_str)
    command = command_str.encode()

    return command


def command_builder_forward_2(point, elev, a='0.05',v='0.15'):
    r_x = str((-point[0] + 0.03)) 
    r_y = str((point[2] -0.15)) 
    r_z = str(point[1] -0.03)
    elev = str(elev)
    command_str = 'movel(pose_trans(' + 'get_forward_kin()' + ',' + 'p[' + r_x + ',' + r_y + ',' + r_z + ',' + '0,0,0]' + '),a=' + a + ',v=' + v + ') \n'
    print(command_str)
    command = command_str.encode()

    return command

def command_builder_forward_3( a='0.05',v='0.15'):
    r_x = str(0) 
    r_y = str(0.1) 
    r_z = str(0)
    command_str = 'movel(pose_trans(' + 'get_forward_kin()' + ',' + 'p[' + r_x + ',' + r_y + ',' + r_z + ',' + '0,0,0]' + '),a=' + a + ',v=' + v + ') \n'
    print(command_str)
    command = command_str.encode()

    return command


def command_builder_qr(x,y, a='0.05',v='0.15'):
    z_move = ((450-y)*0.025232117)/100
    x_move = ((468-x)*0.025238305)/100 #473 previously
    r_x = str(x_move) 
    r_y = str(0)
    r_z = str(-z_move)
    command_str = 'movel(pose_trans(' + 'get_forward_kin()' + ',' + 'p[' + r_x + ',' + r_y + ',' + r_z + ',' + '0,0,0]' + '),a=' + a + ',v=' + v + ') \n'
    print(command_str)
    command = command_str.encode()

    return command

def min_2(point1, point2, point3, point4): #picks out the minimum 2 values of the x and y coordinates
    points_list = [point1, point2, point3, point4]
    min_point = min(points_list)
    points_list.remove(min_point)
    min_point2 = min(points_list)
    return min_point, min_point2


def max_2(point1, point2, point3, point4): #picks out the maximum 2 values of the x and y coordinates
    points_list = [point1, point2, point3, point4]
    max_point = max(points_list)
    points_list.remove(max_point)
    max_point2 = max(points_list)
    return max_point, max_point2

class CV_Pipe:

    def __init__(self, realsense_id):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(realsense_id)

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Camera pipeline functions
    def start(self):
        print('Starting camera')
        profile = self.pipeline.start(self.config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        return

    def update_aligned_frames(self):
        # Wait for a coherent pair of frames: depth and color  
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        self.depth_frame = aligned_frames.get_depth_frame()
        self.color_frame = aligned_frames.get_color_frame()

        # Convert images to numpy arrays
        self.depth_image = np.asanyarray(self.depth_frame.get_data())
        self.color_image = np.asanyarray(self.color_frame.get_data())

        self.intrin = self.depth_frame.profile.as_video_stream_profile().intrinsics
        return

    def output_image(self):
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = self.color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(self.color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((self.color_image, depth_colormap))

        return images

    def output_image_bytes(self):
        images = self.output_image()
        retval, buffer = cv2.imencode('.jpg', images)
        return buffer.tobytes()

    # CV pipeline functions
    def detect_circles_lastmile(self, minRadius=MIN_RADIUS_LASTMILE, maxRadius=MAX_RADIUS_LASTMILE):
        gray_img = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)    
        gray_img = cv2.GaussianBlur(gray_img, (7, 7), 1.5, 1.5);
        circles = cv2.HoughCircles(gray_img,cv2.HOUGH_GRADIENT,1,20,param1=25,param2=20,minRadius=minRadius,maxRadius=maxRadius)
        if circles is not None:
            print('Detected ' + str(circles.shape[1]) + ' circles')
            for i in circles[0,:]:
                i = i.astype(int)
                # draw the outer circle
                print(i[2])
                self.draw_circle((i[0], i[1]), i[2], (0,255,0), 2)
                # draw the center of the circle
                self.draw_circle((i[0], i[1]), 2, (0,0,255), 3)
            self.circles = circles
            self.points = circles[0, :, :-1]
            return circles.shape[1]
        else:
            print('No circles detected')
            return 0

    def detect_qr(self, minRadius=MIN_RADIUS_LASTMILE, maxRadius=MAX_RADIUS_LASTMILE):
        gray_img = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)    
        gray_img = cv2.GaussianBlur(gray_img, (7, 7), 1.5, 1.5);

    def detect_circles_stationary(self, minRadius=MIN_RADIUS_STATIONARY, maxRadius=MAX_RADIUS_STATIONARY):
        gray_img = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)    
        gray_img = cv2.GaussianBlur(gray_img, (7, 7), 1.5, 1.5);
        circles = cv2.HoughCircles(gray_img,cv2.HOUGH_GRADIENT,1,20,param1=25,param2=20,minRadius=minRadius,maxRadius=maxRadius)
        if circles is not None:
            print('Detected ' + str(circles.shape[1]) + ' circles')
            for i in circles[0,:]:
                i = i.astype(int)
                # draw the outer circle
                print(i[2])
                self.draw_circle((i[0], i[1]), i[2], (0,255,0), 2)
                # draw the center of the circle
                self.draw_circle((i[0], i[1]), 2, (0,0,255), 3)
            self.circles = circles
            self.points = circles[0, :, :-1]
            return circles.shape[1]
        else:
            print('No circles detected')
            return 0


    def detect_circles(self, minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS):
        gray_img = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)    
        gray_img = cv2.GaussianBlur(gray_img, (7, 7), 1.5, 1.5);
        circles = cv2.HoughCircles(gray_img,cv2.HOUGH_GRADIENT,1,120,param1=25,param2=20,minRadius=minRadius,maxRadius=maxRadius)
        if circles is not None:
            print('Detected ' + str(circles.shape[1]) + ' circles')
            for i in circles[0,:]:
                i = i.astype(int)
                # draw the outer circle
                print(i[2])
                self.draw_circle((i[0], i[1]), i[2], (0,255,0), 2)
                # draw the center of the circle
                self.draw_circle((i[0], i[1]), 2, (0,0,255), 3)
            self.circles = circles
            self.points = circles[0, :, :-1]
            return circles.shape[1]
        else:
            print('No circles detected')
            return 0

    def update_real_points(self):
        real_points = []
        # Find real points
        check_zero=0
        for point in self.points:
            point = point.astype(int)
            depth = self.depth_frame.as_depth_frame().get_distance(point[0], point[1])
            real_point = rs.rs2_deproject_pixel_to_point(self.intrin, [point[0], point[1]], depth)
            real_points.append(real_point)
            print(real_point)
            for i in range(3):
                if real_point[i]==0:
                    check_zero=1
        self.real_points = real_points
        return(check_zero)

    def update_real_points_lastmile(self):
        real_points = []
        # Find real points
        check_zero=0
        for point in self.points:
            point = point.astype(int)
            print('point')
            print(point)
            real_point = rs.rs2_deproject_pixel_to_point(self.intrin, [point[0], point[1]], 0.14)
            real_points.append(real_point)
            print('real_point')
            print(real_point)
            for i in range(1):
                if real_point[i]==0:
                    check_zero=1
        self.real_points = real_points
        return(check_zero)

    def update_convex_hull(self):
        # Find convex hull of detected circles
        self.hull = ConvexHull(self.points)
        self.draw_polygon(self.hull.simplices)

    def update_delaunay(self):        
        # Break polygon down to triangles
        self.vertices = self.points[self.hull.vertices]
        self.tri = Delaunay(self.vertices)

    def gen_points_in_polygon(self, num_points, rng=RNG):
        sample_points = []
        for idx_simplex, simplex in enumerate(self.tri.simplices):
            tri_points = self.vertices[simplex].astype(int)
            self.draw_triangles(tri_points)
            sample_points.extend(self.gen_points_in_triangle(tri_points, idx_simplex, int(num_points/self.tri.simplices.size), rng))
        self.sample_points = sample_points

    def gen_points_in_triangle(self, tri_points, idx_simplex, num_points, rng):
        sample_points = []
        for i in range(num_points):
            too_close = False
            r1 = RNG.random()
            r2 = RNG.random()
            gen_point = ((1 - sqrt(r1)) * tri_points[0]) + ((sqrt(r1) * (1-r2)) * tri_points[1]) + ((r2 * sqrt(r1)) * tri_points[2])
            gen_point = gen_point.astype(int)
            for vertex in tri_points:
                if self.distance_two_points(vertex, gen_point) < MAX_RADIUS:
                    too_close = True
            if not too_close:
                self.draw_circle(tuple(gen_point), 2, (255,255,0), 1)
                depth = self.depth_frame.as_depth_frame().get_distance(gen_point[0], gen_point[1])
                point = rs.rs2_deproject_pixel_to_point(self.intrin, [gen_point[0], gen_point[1]], depth)
                sample_points.append(point)
        print('sample_points')
        print(sample_points)
        return sample_points

    def stop(self):
        print('Stopping camera')
        self.pipeline.stop()
        return

    # Spatial fuctions
    def best_fit_plane(self):
        self.plane = Plane.best_fit(Points(self.sample_points))
        if self.plane.normal[2] < 0:
            self.plane.normal = -self.plane.normal
        return self.plane.normal

    def find_rotation_stationary(self):
        tilt = atan(self.plane.normal[0]/self.plane.normal[2])
        elev = atan(self.plane.normal[1]/self.plane.normal[2])
        print(tilt / (pi/180))
        print(elev / (pi/180))
        print("elev")
        print(elev)
        #elev = pi + elev
        #self.rot = R.from_euler('XZX',[-pi/2,tilt,0])
        self.rot = R.from_euler('YZY',[pi/2,tilt,0])

        print('Euler angles')
        print(self.rot.as_euler('XZX',degrees=True))
        print('Rotation vector')
        print(self.rot.as_rotvec())
        print('Euler YZY')
        print(self.rot.as_euler('YZY',degrees=True))
        return(self.rot.as_euler('XZX',degrees=True))

    def find_rotation(self):
        tilt = atan(self.plane.normal[0]/self.plane.normal[2])
        elev = atan(self.plane.normal[1]/self.plane.normal[2])
        #elev = pi + elev
        #self.rot = R.from_euler('XZX',[-pi/2,tilt,0])
        self.rot = R.from_euler('XZX',[elev,0,0])

        print('Euler angles')
        print(self.rot.as_euler('YZY',degrees=True))
        print('Rotation vector')
        print(self.rot.as_rotvec())
        print("elev")
        print(elev)
        return(elev)

    # Drawing functions
    def draw_circle(self, point, radius, color=(255,255,0), thickness=1):
        cv2.circle(self.color_image, point, radius, color, thickness)

    def draw_polygon(self, simplices, color=(255,255,255), thickness=2):
        for simplex in simplices:
            start_point = tuple(self.points[simplex][0].astype(int))
            end_point = tuple(self.points[simplex][1].astype(int))
            cv2.line(self.color_image, start_point, end_point, color, thickness)

    def draw_triangles(self, tri_points, color=(255,255,255), thickness=2):
        cv2.line(self.color_image, tuple(tri_points[0]), tuple(tri_points[1]), color, thickness)
        cv2.line(self.color_image, tuple(tri_points[1]), tuple(tri_points[2]), color, thickness)
        cv2.line(self.color_image, tuple(tri_points[2]), tuple(tri_points[0]), color, thickness)

    # Utility functions
    def distance_two_points(self, point1, point2):
        return np.linalg.norm(point1-point2)

if __name__ == '__main__':
    try:
#######start stationary camera#######################
        pipe = CV_Pipe('040322070700')
        pipe.start()
        
        num_circles = 0
        while num_circles != NUM_CIRCLES:
            pipe.update_aligned_frames()

            # Detect circles
            num_circles = pipe.detect_circles_stationary()

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', pipe.output_image())
            cv2.waitKey(1)

        cv2.imshow('RealSense', pipe.output_image())
        cv2.waitKey(0)
        check_zero=pipe.update_real_points()
        pipe.update_convex_hull()
        pipe.update_delaunay()
        pipe.gen_points_in_polygon(NUM_POINTS, RNG)

        while (check_zero==1):
            pipe = CV_Pipe('040322070700')
            pipe.start()
        
            num_circles = 0
            while num_circles != NUM_CIRCLES:
                pipe.update_aligned_frames()

                # Detect circles
                num_circles = pipe.detect_circles()

                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', pipe.output_image())
                cv2.waitKey(1)

            cv2.imshow('RealSense', pipe.output_image())
            cv2.waitKey(0)

            check_zero = pipe.update_real_points()
            pipe.update_convex_hull()
            pipe.update_delaunay()
            pipe.gen_points_in_polygon(NUM_POINTS, RNG)


        print('normal')
        pipe.best_fit_plane()
        print(pipe.plane.normal)
        pipe.find_rotation_stationary()


        pipe.stop()

        s = connect_to_robot()

        min_x = min_2(pipe.real_points[0][0], pipe.real_points[1][0], pipe.real_points[2][0], pipe.real_points[3][0]) # to get the bottom left hole
        min_y = min_2(pipe.real_points[0][1], pipe.real_points[1][1], pipe.real_points[2][1], pipe.real_points[3][0])

        for i in range(4):
            if ((pipe.real_points[i][0] in min_x) and (pipe.real_points[i][1] in min_y)):
                response = send_command(s,command_builder(pipe.real_points[i],pipe.rot.as_rotvec(),pipe.rot.as_euler('XZX',degrees=True)))
                time.sleep(10)
                print(response)
                break

########start end effector camera#########
        for j in range(4):
            pipe = CV_Pipe('040322073606')
            pipe.start()
            

            num_circles = 0
            while num_circles != NUM_CIRCLES:
                pipe.update_aligned_frames()

                # Detect circles
                num_circles = pipe.detect_circles()

                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', pipe.output_image())
                cv2.waitKey(1)

            cv2.imshow('RealSense', pipe.output_image())
            cv2.waitKey(0)
            check_zero = pipe.update_real_points()
            print('check_zero')
            print(check_zero)
            print("update_real_points")
            print(pipe.update_real_points())
            pipe.update_convex_hull()
            pipe.update_delaunay()
            pipe.gen_points_in_polygon(NUM_POINTS, RNG)

            print('normal')
            pipe.best_fit_plane()
            print(pipe.plane.normal)
            pipe.find_rotation()

            while (check_zero==1):
                pipe = CV_Pipe('040322073606')
                pipe.start()
            
                num_circles = 0
                while num_circles != NUM_CIRCLES:
                    pipe.update_aligned_frames()

                    # Detect circles
                    num_circles = pipe.detect_circles()

                    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('RealSense', pipe.output_image())
                    cv2.waitKey(1)

                cv2.imshow('RealSense', pipe.output_image())
                cv2.waitKey(0)
                check_zero = pipe.update_real_points()
                pipe.update_convex_hull()
                pipe.update_delaunay()
                pipe.gen_points_in_polygon(NUM_POINTS, RNG)

                print('normal')
                pipe.best_fit_plane()
                print(pipe.plane.normal)
                pipe.find_rotation()
                
            elev=pipe.find_rotation()
            print("elev")
            print(elev)

            pipe.stop()

            max_x = max_2(pipe.real_points[0][0], pipe.real_points[1][0], pipe.real_points[2][0], pipe.real_points[3][0])
            max_y = max_2(pipe.real_points[0][1], pipe.real_points[1][1], pipe.real_points[2][1], pipe.real_points[3][1])
            min_x = min_2(pipe.real_points[0][0], pipe.real_points[1][0], pipe.real_points[2][0], pipe.real_points[3][0])
            min_y = min_2(pipe.real_points[0][1], pipe.real_points[1][1], pipe.real_points[2][1], pipe.real_points[3][1])

            if j==0: #go to attachment #1
                for i in range(4):
                    if ((pipe.real_points[i][0] in max_x) and (pipe.real_points[i][1] in max_y)):
                        response = send_command(s,command_builder_rotate(pipe.real_points[i],elev))
                        time.sleep(5)
                        response = send_command(s,command_builder_forward_2(pipe.real_points[i],elev))
                        time.sleep(10)

                        break

            elif j==1: #go to attachment #2
                for i in range(4):
                    if ((pipe.real_points[i][0] in min_x) and (pipe.real_points[i][1] in min_y)):
                        response = send_command(s,command_builder_forward_2(pipe.real_points[i],0))
                        time.sleep(10)
                        break

            elif j==2: #go to attachment #3                      
                for i in range(4):
                    if ((pipe.real_points[i][0] in max_x) and (pipe.real_points[i][1] not in max_y)):
                        response = send_command(s,command_builder_forward_2(pipe.real_points[i],0))
                        time.sleep(10)
                        break             

            elif j==3: #go to attachment #4
                for i in range(4):
                    if ((pipe.real_points[i][0] not in max_x) and (pipe.real_points[i][1] in max_y)):
                        print(pipe.real_points[i])
                        response = send_command(s,command_builder_forward_2(pipe.real_points[i],0))
                        time.sleep(10)
                        break

            pipe = CV_Pipe('040322073606')
            pipe.start()

            
            num_circles = 0
            result = []
            while result ==[]:
                pipe.update_aligned_frames()

                # Detect circles
                num_circles = pipe.detect_qr() ### changed this

                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                pipe.update_aligned_frames()

                #Apr Tag 
                save_image = Image.fromarray(pipe.color_image).save('AprTag.png')
                Image.open("AprTag.png").save("AprTag.jpg")
                aprtag_image = cv2.imread("AprTag.jpg")
                aprtag_image_gray = cv2.cvtColor(aprtag_image, cv2.COLOR_BGR2GRAY)

                options = apriltag.DetectorOptions(families="tag36h11")
                detector = apriltag.Detector(options)
                result = detector.detect(aprtag_image_gray)

            print("result")
            print(result[0].center[0],result[0].center[1])
            response = send_command(s,command_builder_qr(result[0].center[0],result[0].center[1]))
            time.sleep(5)
            response = send_command(s,command_builder_forward_3())
            time.sleep(5)
        
            if j==0:
                response = send_command(s,command_builder_back())
                time.sleep(5)
                response = send_command(s,command_builder_adjust(0,0.05))
                time.sleep(5)        

            if j==1:              
                response = send_command(s,command_builder_back())
                time.sleep(5)
                response = send_command(s,command_builder_adjust(0.1,-0.13))
                time.sleep(5)   

            if j==2:         
                response = send_command(s,command_builder_back())
                time.sleep(5)
                response = send_command(s,command_builder_adjust(0.1,0.1))
                time.sleep(5)

            if j==3:  
                response = send_command(s,command_builder_back())
                time.sleep(5)



        pipe.stop()
    


# #last mile
#             pipe = CV_Pipe('040322073606')
#             pipe.start()
            
#             num_circles = 0
#             while num_circles != 1: #NUM_CIRCLES
#                 pipe.update_aligned_frames()

#                 # Detect circles
#                 num_circles = pipe.detect_circles_lastmile()

#                 cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#                 cv2.imshow('RealSense', pipe.output_image())
#                 cv2.waitKey(1)

#             cv2.imshow('RealSense', pipe.output_image())
#             cv2.waitKey(0)
#             check_zero=pipe.update_real_points_lastmile()

#             while (check_zero==1):
#                 pipe = CV_Pipe('040322073606')
#                 pipe.start()
            
#                 num_circles = 0
#                 while num_circles != 1:
#                     pipe.update_aligned_frames()

#                     # Detect circles
#                     num_circles = pipe.detect_circles_lastmile()

#                     cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#                     cv2.imshow('RealSense', pipe.output_image())
#                     cv2.waitKey(1)
                    

#                 cv2.imshow('RealSense', pipe.output_image())
#                 cv2.waitKey(0)
#                 check_zero=pipe.update_real_points_lastmile()

#             pipe.stop()

#             if j==0:
#                 response = send_command(s,command_builder_forward_3(pipe.real_points[0],elev))
#                 time.sleep(5)
#                 response = send_command(s,command_builder_forward_4(pipe.real_points[0],elev))
#                 time.sleep(5)                
#                 response = send_command(s,command_builder_back(pipe.real_points[0]))
#                 time.sleep(5)
#                 response = send_command(s,command_builder_adjust(pipe.real_points[0],0,0.05))
#                 time.sleep(5)        

#             if j==1:
#                 response = send_command(s,command_builder_forward_3(pipe.real_points[0],elev))
#                 time.sleep(5)
#                 response = send_command(s,command_builder_forward_4(pipe.real_points[0],elev))
#                 time.sleep(5)                
#                 response = send_command(s,command_builder_back(pipe.real_points[0]))
#                 time.sleep(5)
#                 response = send_command(s,command_builder_adjust(pipe.real_points[0],0.1,-0.13))
#                 time.sleep(5)   

#             if j==2:         
#                 response = send_command(s,command_builder_forward_3(pipe.real_points[0],elev))
#                 time.sleep(5)
#                 response = send_command(s,command_builder_forward_4(pipe.real_points[0],elev))
#                 time.sleep(5) 
#                 response = send_command(s,command_builder_back(pipe.real_points[0]))
#                 time.sleep(5)
#                 response = send_command(s,command_builder_adjust(pipe.real_points[0],0.1,0.1))
#                 time.sleep(5)

#             if j==3:
#                 response = send_command(s,command_builder_forward_3(pipe.real_points[0],elev))
#                 time.sleep(5)
#                 response = send_command(s,command_builder_forward_4(pipe.real_points[0],elev))
#                 time.sleep(5)    
#                 response = send_command(s,command_builder_back(pipe.real_points[0]))
#                 time.sleep(5)


        cv2.imshow('RealSense', pipe.output_image())
        cv2.waitKey(0)



    except Exception as e:
        print(e)
        print(traceback.format_exc())
        pipe.stop()
