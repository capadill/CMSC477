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


def movePath(path):
    print("Beginning Navigation")
    queue = path
   # queue[0] = 60
    print(queue)
    current_node = int(queue.pop(0))

    while queue:
        goal_node = int(queue.pop(0))#pop takes element out off the array
        #nodes must be neighbors
        direction = current_node - goal_node
        print(direction)
        #if positive, moving down/right (50-40)
        #if negative, moving up/left (50-60)
        #if difference of 10, moving up/down
        #if difference of 1, moving right/left
        #if difference of 9, negative slope diagonal
        #if difference of 11, positive slope diagonal
        #use all of these to create a decision tree
        if direction < 0:
            if direction == -1: #moving right
                print("Moving Right...")
                ep_chassis.move(x=.425, y=-0, z=0, xy_speed=0.6).wait_for_completed()
            elif direction == -10: #moving up
                print("Moving Up...")
                ep_chassis.move(x=0, y=-.45, z=0, xy_speed=0.6).wait_for_completed()
            elif direction == -11: #moving up & right
                print("Moving Up & Right...")
                ep_chassis.move(x=.266, y=-.266, z=0, xy_speed=0.6).wait_for_completed()
            elif direction == -9: #moving up & left
                print("Moving Up & Left...")
                ep_chassis.move(x=-.266, y=.266, z=0, xy_speed=0.6).wait_for_completed()
        if direction > 0:
            if direction == 1: #moving left
                print("Moving Left...")
                ep_chassis.move(x=-.425, y=0, z=0, xy_speed=0.6).wait_for_completed()
            elif direction == 10: #moving down
                print("Moving Down...")
                ep_chassis.move(x=0, y=.45, z=0, xy_speed=0.6).wait_for_completed()
            elif direction == 11: #moving down & left
                print("Moving Down & Left...")
                ep_chassis.move(x=.266, y=.266, z=0, xy_speed=0.6).wait_for_completed()
            elif direction == 9: #moving down & right
                print("Moving Down & Right...")
                ep_chassis.move(x=.266, y=.266, z=0, xy_speed=0.6).wait_for_completed()
        current_node = goal_node
        stop=0

        if goal_node == 32:
            #key = res.tag_id
            print("apritag 5")
            while stop==0:
                    try:
                        img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)  
                        cv2.imwrite("/home/user/Desktop/test.png", img) 
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        gray.astype(np.uint8)

                        K=np.array([[184.752, 0, 320], [0, 184.752, 180], [0, 0, 1]])

                        results = at_detector.detect(gray, estimate_tag_pose=False)
                        for res in results:
                            print(res.tag_id)
                            pose = find_pose_from_tag(K, res)
                            rot, jaco = cv2.Rodrigues(pose[1], pose[1])
                            print(pose)
                            if res.tag_id==0:
                                print("Here!!!")
                                stop=1
                        

                            #pose = find_pose_from_tag(K, res)
                            #rot, jaco = cv2.Rodrigues(pose[1], pose[1])
                            #print(rot)

                            pts = res.corners.reshape((-1, 1, 2)).astype(np.int32)
                            img = cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=5)
                            cv2.circle(img, tuple(res.center.astype(np.int32)), 5, (0, 0, 255), -1)

                            #print(pose)

                        cv2.imshow("img", img)
                        cv2.waitKey(10)
                    except KeyboardInterrupt:
                        ep_camera.stop_video_stream()
                        ep_robot.close()
                        print ('Exiting')
                        exit(1)

                                    
                

            
        
            
    return

def bfs(graph, start, goal): #FIFO means if you've been waiting the longest, you get priority
    queue = [(start, [start])]
    visited = set()
    
    while queue:
        print(queue)
        current_node,path = queue.pop(0)

        if current_node not in visited:
            visited.add(current_node)
            print("Current Node: ", current_node) #action goes here

            if current_node == goal:
                return path


            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    return path


# Example graph representation as an adjacency list
graph = {
    '00': ['10','11','01'],
    '01': ['00','10','11','12','02'],
    '02': ['01','11','12','13','03'],
    '03': ['02','12','13','14','04'],
    '04': ['03','13','14'],
    #05
    '06': ['16','17','07'],
    '07': ['06','16','17','18','08'],
    '08': ['07','17','18','19','09'],
    '09': ['08','18','19','110','010'],
    '010': ['09','19','110'],
    '10': ['00','01','11','21','20'],
    '11': ['00','10','20','21','22','12','02','01'],
    '12': ['01','11','21','22','23','13','03','02'],
    '13': ['02','12','22','23','24','14','04','03'],
    '14': ['04','03','13','23','24'],
    #15
    '16': ['26','27','17','07','06'],
    '17': ['26','27','28','18','08','07','06','16'],
    '18': ['27','28','29','19','09','08','07','17'],
    '19': ['28','29','210','110','010','09','08','18'],
    '110': ['010','09','19','29','210'],
    '20': ['10','11','21','31','30'],
    '21': ['10','20','30','31','32','22','12','11'],
    '22': ['11','21','31','32','33','23','13','12'],
    '23': ['12','22','32','33','34','24','14','13'],
    '24': ['14','13','23','33','34'],
    #25
    '26': ['16','17','27','37','36'],
    '27': ['16','26','36','37','38','28','18','17'],
    '28': ['17','27','37','38','39','29','19','18'],
    '29': ['18','28','38','39','310','210','110','19'],
    '210': ['19','29','39','310','110'],
    '30': ['20','21','31','41','40'],
    '31': ['20','30','40','41','32','22','21'],
    '32': ['21','31','33','23','22'],
    '33': ['22','32','43','44','34','24','23'],
    '34': ['23','33','43','44','24'],
    #35
    '36': ['46','47','37','27','26'],
    '37': ['26','36','46','47','38','28','27'],
    '38': ['27','37','39','29','28'],
    '39': ['28','38','49','410','310','210','29'],
    '310': ['29','39','49','410','210'],
    '40': ['50','51','41','31','30'],
    '41': ['30','40','50','51','31'],
    #42
    '43': ['32','53','54','44','34','33'],
    '44': ['23','33','43','44','24'],
    #45
    '46': ['56','57','47','37','36'],
    '47': ['36','46','56','57','37'],
    #48
    '49': ['59','510','410','310','39'],
    '410': ['39','49','59','510','310'],
    '50': ['60','61','51','41','40'],
    '51': ['40','50','60','61','41'],
    #52
    '53': ['63','64','54','44','43'],
    '54': ['43','53','63','64','65','55','44'],
    '55': ['44','54','64','65','66','56'],
    '56': ['55','65','66','67','57','47','46'],
    '57': ['46','56','66','67','47'],
    #58
    '59': ['49','69','610','510','410'],
    '510': ['49','59','69','610','410'],
    '60': ['50','70','71','61','51'],
    '61': ['50','60','70','71','51'],
    #62
    '63': ['73','74','64','54','53'],
    '64': ['53','63','73','74','75','65','55','54'],
    '65': ['54','64','74','75','76','66','56','55'],
    '66': ['55','65','75','76','77','67','57','56'],
    '67': ['56','66','76','77','57'],
    #68
    '69': ['79','710','610','510','59'],
    '610': ['59','69','79','710','510'],
    '70': ['60','80','81','71','61'],
    '71': ['60','70','80','81','61'],
    #72
    '73': ['74','64','63'],
    '74': ['63','73','75','65','64'],
    '75': ['64','74','76','66','65'],
    '76': ['65','75','77','67','66'],
    '77': ['66','76','67'],
    #78
    '79': ['89','810','710','610','69'],
    '710': ['69','79','89','810','610'],
    '80': ['81','71','70'],
    '81': ['70','80','71'],
    #82
    #83
    #84
    #85
    #86
    #87
    #88
    '89': ['810','710','79'],
    '810': ['79','89','710']

}
#init
ep_robot = robot.Robot()
ep_robot.initialize(conn_type="ap")

ep_chassis = ep_robot.chassis
ep_camera = ep_robot.camera
ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
tag_size=0.16# only need this line of code if I use "find_pose_from_tag" code
"""" while stop==0:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)  
            cv2.imwrite("/home/user/Desktop/test.png", img) 
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray.astype(np.uint8)

            K=np.array([[184.752, 0, 320], [0, 184.752, 180], [0, 0, 1]])

            results = at_detector.detect(gray, estimate_tag_pose=False)
            for res in results:
                print(res.tag_id)
                if res.tag_id==0:
                    print("Here!!!")
                    stop=1
            

                #pose = find_pose_from_tag(K, res)
                #rot, jaco = cv2.Rodrigues(pose[1], pose[1])
                #print(rot)

                pts = res.corners.reshape((-1, 1, 2)).astype(np.int32)
                img = cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=5)
                cv2.circle(img, tuple(res.center.astype(np.int32)), 5, (0, 0, 255), -1)

                #print(pose)

            cv2.imshow("img", img)
            cv2.waitKey(10)
        

        except KeyboardInterrupt:
            ep_camera.stop_video_stream()
            ep_robot.close()
            print ('Exiting')
            exit(1) """


start_node = '50'
goal_node = '510'
path_to_goal = bfs(graph, start_node, goal_node)
if path_to_goal:
    print("Path to goal:", path_to_goal)
    movePath(path_to_goal)
else:
    print("Goal not found.")
