from ultralytics import YOLO
import cv2 
import numpy as np
import time
from robomaster import robot
from robomaster import camera
import matplotlib.pyplot as plt

def find_and_plot_blue_line(image):
    # Load the image
    original_image = image.copy()  # Make a copy to preserve the original
    # Convert the image to HSV color space
    img = np.array(image)
    
    # Define range of blue color in HSV
    lower_blue = np.array([70,68,90])
    upper_blue = np.array([200,255,255])
    cc = 0 # Lower image snip bound
    aa = 100 # Upper image snip bound
    snip = np.zeros((img.shape[0], img.shape[1]), dtype ="uint8")
    snip = img[(img.shape[0] - aa):(img.shape[0] - cc), (0):(img.shape[1])]

    hsv = cv2.cvtColor(snip, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours in the blue line image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Initialize variables to store information about the longest contour
        longest_contour = None
        max_contour_length = 0
    
    # Iterate through all contours to find the longest one
        longest_contour = contours[0]
        for contour in contours:
            # Calculate the length of the current contour
            contour_length = cv2.arcLength(contour, True)
            
            # Check if the current contour is longer than the previous longest contour
            if contour_length > max_contour_length:
                max_contour_length = contour_length
                longest_contour = contour

        vx, vy, x, y = cv2.fitLine(longest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        contour_length = cv2.arcLength(longest_contour, True)
        # Calculate slope
        slope = vy / vx
        
        # Get two points to draw the line
        left_point = (int(x - 1000 * vx), int(y - 1000 * vy))
        right_point = (int(x + 1000 * vx), int(y + 1000 * vy))
        
        # Plot the line on the original image
        line_image = cv2.line(snip, left_point, right_point, (0, 255, 0), 2)
        
        #cv2.imshow("Mask",mask)
        # cv2.imshow("Line",line_image)
        # cv2.waitKey(10)
        
        return slope[0], y[0], contour_length
    
    else:
        print("No blue line found in the image.")
        return 0, 0, 0

def sub_data_handler(sub_info):
    pos_x, pos_y = sub_info
    global XP
    global YP
    #print("Robotic Arm: pos x:{0}, pos y:{1}".format(pos_x, pos_y))
    XP = pos_x
    YP = pos_y

def moving_avg(y_data, pt):
    data = [None]*len(y_data)
    for i in range(len(y_data)):
        avg = 0
        count = 0
        for j in range(pt):
            if i-j >= 0:
                avg += y_data[i-j]
                count += 1

        data[i] = avg/count
    return data[-1]

def box_distance(dist, w, h):
    # Determine constants through experimentation
    # m is Z and b is K
    Zw = 100
    Kw = 0
    Zh = 100
    Kh = 0
    w_distance = Zw/w + Kw
    h_distance = Zh/h + Kh
    distance = (w_distance + h_distance)/2
    dist.append(distance)
    pts = 3 # amount of points considered in moving average (current + (pts-1)*past_points)
    current_distance = moving_avg(dist, pts)
    return current_distance

def ResetArm():
    # Gripper Reset
    ep_gripper.open(power=50)
    time.sleep(1)
    ep_gripper.pause()
    # Arm Reset
    ep_arm.sub_position(freq=5, callback=sub_data_handler)
    time.sleep(1)
    x1 = 80
    y1 = 75
    ep_arm.move(x=(x1-XP)-5).wait_for_completed()
    ep_arm.move(y=(y1-YP)-5).wait_for_completed()
    ep_arm.move(y=-10).wait_for_completed()
    time.sleep(1)
    ep_arm.unsub_position()

def FindTower(model):
    # rotate until blue line is found with high enough confidence
    TowerFound = False
    z_val = 40
    while not TowerFound:
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=5)  
        #cv2.imshow("Frame",frame)
        #cv2.waitKey(10)
        results = model.predict(source=frame, conf=0.8)

        if len(results[0]) > 0:
            TowerFound = True
        else:
            ep_chassis.drive_speed(x=0, y=0, z=z_val, timeout=5)
            
    # Stop spinning
    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)

def Go2Tower(model):
    
    # constants
    dist = []
    target_distance = 1.6 # arbitrary needs to be tuned
    distance_tol = 0.07
    box_offset_tol = 3 # number of pixels left or right of the center
    ON_TARGET = False
    while(not ON_TARGET): 
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=5)  
        img = np.array(frame)
        #cv2.imshow("Frame",frame)
        #cv2.waitKey(10)
        results = model.predict(source=frame, conf=0.7)

        conf = results[0].boxes.conf
        conf = conf.cpu()
        if len(conf)>0:
            best = np.argmax(conf)
        else:
            continue
        coords = results[0].boxes.xywh
        # finds the coordinates of the bounding box with the highest confidence
        c = coords[best]
        x_center = int(img.shape[1]/2)
        x = int(c[0])
        y = int(c[1])
        w = int(c[2])
        h = int(c[3])
        box_offset = x_center - x
        distance = box_distance(dist, w, h)
        if (abs(box_offset) > box_offset_tol) and not ON_TARGET:
            Kz = -0.15
            z_val = box_offset*Kz
            ep_chassis.drive_speed(x=0, y=0, z=z_val, timeout=5)
            # turn till box is in the center
        elif(abs(distance - target_distance) >= distance_tol) and not ON_TARGET :
            Kx = 0.175
            x_val = (distance - target_distance)*Kx
            print(distance)
            ep_chassis.drive_speed(x=x_val, y=0, z=0, timeout=5)
            # move till box is at target distance
        elif(abs(distance - target_distance) < distance_tol):
            ON_TARGET = True
            print("done")
            x_val = 0.21
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
            ep_chassis.move(x=x_val, y=0, z=0, xy_speed=0.7).wait_for_completed()
            # move x amount

    cv2.destroyAllWindows()

def PickupTower():
    # use arm commands to grab and pickup the tower
    ep_arm.sub_position(freq=5, callback=sub_data_handler)
    time.sleep(1)
    x2 = 150
    y2 = 20
    x3 = 190
    y3 = 50
    # Move Arm
    ep_arm.move(x=(x2-XP)+5).wait_for_completed()
    ep_arm.move(y=(y2-YP)-5).wait_for_completed()
    ep_arm.move(x=(x3-XP)+5).wait_for_completed()
    # Close Gripper
    ep_gripper.close(power=100)
    time.sleep(2.5)
    ep_gripper.pause()
    # Move Arm Up a Little
    ep_arm.move(y=(y3-YP)+5).wait_for_completed()
    time.sleep(0.5)
    ep_arm.unsub_position()

def FindBlueLine():
    # rotate until blue line is found with high enough confidence
    LineFound = False
    z_val = 10
    min_cont_len = 300
    while not LineFound:
        img = ep_camera.read_cv2_image(strategy="newest", timeout=5)  
        slope, y, cont_len = find_and_plot_blue_line(img)
        if cont_len < min_cont_len:
            ep_chassis.drive_speed(x=0, y=0, z=z_val, timeout=5)
        else:
            LineFound = True
    # Stop spinning
    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
    
def Go2BlueLine():
    # move close to the blue line and straighten out
    angle_tol = 0.01
    Kz = 100 #
    y_Goal = 100
    y = y_Goal
    y_tol = 10
    straight = False
    done = False
    # Align with blue line
    while not done:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=5)  
            slope, y, cont_size = find_and_plot_blue_line(img)
            y_dist = y_Goal-y
            if slope == None:
                FindBlueLine()
            if (np.abs(slope) > angle_tol) and not straight:
                z_val = Kz*slope
                ep_chassis.drive_speed(x=0, y=0, z=z_val, timeout=5)
            elif not straight: 
                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
                ep_chassis.move(x=0, y=0, z=0, xy_speed=0.7).wait_for_completed()
                straight = True
                print("Bam bidam")
            elif straight and not done:
                print(y_dist)
                if abs(y_dist) > y_tol:
                    Kx = 0.005
                    x_val = Kx*y_dist
                    if (np.abs(slope) > angle_tol):
                        z_val = 0.5*Kz*slope
                    else: 
                        z_val = 0
                    ep_chassis.drive_speed(x=x_val, y=0, z=z_val, timeout=5)
                else: 
                    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
                    x_dist = 0.43
                    time.sleep(1)
                    ep_chassis.move(x=x_dist, y=0, z=0, xy_speed=0.7).wait_for_completed()
                    done = True

        except KeyboardInterrupt:
            ep_camera.stop_video_stream()
            ep_robot.close()
            print ('Exiting')
            exit(1)

def Wait():
    time.sleep(12.0)
def LetGo():
    # Gripper Reset
    ep_gripper.open(power=50)
    time.sleep(1)
    ep_gripper.pause()
    x_val = -5
    ep_chassis.move(x=x_val, y=0, z=0, xy_speed=0.7).wait_for_completed()


if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    ep_chassis = ep_robot.chassis
    ep_gripper = ep_robot.gripper
    ep_arm = ep_robot.robotic_arm

    model = YOLO("/home/capadill/CMSC477_ws/best2.pt")
    ResetArm()
    FindTower(model)
    Go2Tower(model)
    PickupTower()
    FindBlueLine()
    Go2BlueLine()
    Wait()
    LetGo()