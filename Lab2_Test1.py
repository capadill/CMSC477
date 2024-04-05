from pupil_apriltags import Detector
import cv2
import numpy as np
import time
from robomaster import robot
from robomaster import camera


at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)
def find_pose_from_tag(K, detection):
    m_half_size = tag_size / 2

    marker_center = np.array((0, 0, 0))
    marker_points = []
    marker_points.append(marker_center + (-m_half_size, m_half_size, 0))
    marker_points.append(marker_center + ( m_half_size, m_half_size, 0))
    marker_points.append(marker_center + ( m_half_size, -m_half_size, 0))
    marker_points.append(marker_center + (-m_half_size, -m_half_size, 0))
    _marker_points = np.array(marker_points)

    object_points = _marker_points
    image_points = detection.corners

    pnp_ret = cv2.solvePnP(object_points, image_points, K, distCoeffs=None,flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if pnp_ret[0] == False:
        raise Exception('Error solving PnP')

    r = pnp_ret[1]
    p = pnp_ret[2]

    return p.reshape((3,)), r.reshape((3,))

def runCamera(tag_num):
    scalar_x = scalar_y = scalar_z = 0
    tag = 100000
    #print("Running runCamera")
    try:
        img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)  
        cv2.imwrite("/home/user/Desktop/test.png", img) 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray.astype(np.uint8)

        K=np.array([[184.752, 0, 320], [0, 184.752, 180], [0, 0, 1]])

        results = at_detector.detect(gray, estimate_tag_pose=False)
        for res in results:
            print("TAG ID: ", res.tag_id)
            pose = find_pose_from_tag(K, res)
            rot, jaco = cv2.Rodrigues(pose[1], pose[1])
            if res.tag_id == tag_num:
                tag = res.tag_id
                scalar_x = (pose[0][2]-.23)/5 #equals distance from ideal spot from marker (.25 m away)
                scalar_y = pose[0][0]/4 #controls displacement left-to-right from marker
                scalar_z = -2*(((pose[1][2]))*180 / np.pi)
            else:
                tag = 1000000
            pts = res.corners.reshape((-1, 1, 2)).astype(np.int32)
            img = cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=5)
            cv2.circle(img, tuple(res.center.astype(np.int32)), 5, (0, 0, 255), -1)

        cv2.imshow("img", img)
        cv2.waitKey(10)
    except KeyboardInterrupt:
        ep_camera.stop_video_stream()
        ep_robot.close()
        print ('Exiting')
        exit(1)
    return scalar_x,scalar_y,scalar_z,tag

def movePath(tag_num):
    stop = 0
    while stop==0:
        #print("Running Centering")
        scalar_x,scalar_y,scalar_z,tag = runCamera(tag_num)
        if scalar_x == scalar_y == scalar_z == 0: #indicates apriltag has been lost
            ep_chassis.move(x=-.1, y=0, z=0, xy_speed=0.75).wait_for_completed()
            time.sleep(.5)
            finder(tag_num) #tries to find it again
            
        if scalar_x < .01 and scalar_x > -.01 and scalar_y < .01 and scalar_y > -.01 and scalar_z < 1 and scalar_z > -1:
            print("Successfully Centered - Exiting")
            stop=1

        if scalar_x < .03 and scalar_x > 0:
            scalar_x = .03
        if scalar_x > -.03 and scalar_x < 0:
            scalar_x = -.03
        if scalar_y < .03 and scalar_y > 0:
            scalar_y = .03
        if scalar_y > -.03 and scalar_y < 0:
            scalar_y = -.03
        if scalar_z < 4 and scalar_z > 0:
            scalar_z = 4
        if scalar_z > -4 and scalar_z < 0:
            scalar_z = -4

        print("Scalar X: ", scalar_x,"Scalar Y: ", scalar_y, "Scalar Z: ", scalar_z)
        ep_chassis.move(x=scalar_x, y=scalar_y, z=scalar_z, xy_speed=0.75).wait_for_completed()
        time.sleep(.5)

    return

def finder(tag_num):
    tag = 1000000
    while tag != tag_num:
        scalar_x,scalar_y,scalar_z,tag = runCamera(tag_num)
        ep_chassis.move(x=0, y=0, z=10, xy_speed=0.75).wait_for_completed()

def robotGrab():
    ep_gripper.open(power=50)
    time.sleep(1)
    ep_gripper.pause()
    ep_arm.move(x=50,y=-30).wait_for_completed()
    ep_gripper.close(power=50)
    time.sleep(1)
    ep_gripper.pause()
    ep_arm.move(x=-50,y=30).wait_for_completed()

def robotRelease():
    ep_arm.move(x=50,y=-30).wait_for_completed()
    ep_gripper.open(power=50)
    time.sleep(1)
    ep_gripper.pause()
    ep_arm.move(x=-50,y=30).wait_for_completed()
    ep_gripper.close(power=50)
    time.sleep(1)

if __name__ == "__main__":
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_gripper = ep_robot.gripper
    ep_arm = ep_robot.robotic_arm
    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    tag_size=0.16# only need this line of code if I use "find_pose_from_tag" code

    finder(8) #Find tag 8
    print("Lego Tag Found")
    movePath(8)
    print("Grabbing Object")
    robotGrab()
    print("Finding Robot")
    finder(15) #Find tag 15
    print("Robot Tag Found")
    movePath(15)
    print("Exchanging Object")
    robotRelease()

    ep_camera.stop_video_stream()
    ep_robot.close()
    print ('Complete. Exiting')
    exit(1)