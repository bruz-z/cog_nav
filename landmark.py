import airsim
import cv2
import numpy as np
import time

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()

# set camera name and image type to request images and detections
camera_name = "0"
image_type = airsim.ImageType.Scene
client.enableApiControl(True) 
# set detection radius in [cm]
client.simSetDetectionFilterRadius(camera_name, image_type, 600 * 300)
# add desired object name to detect in wild card/regex format
client.simAddDetectionFilterMeshName(camera_name, image_type, "Cylinder*")

# Set the destination goal
destination_goal = np.array([50, 50, 0])
theta_to_goal = np.arctan2(destination_goal[1], destination_goal[0])
radius = 15
# Function to navigate to a given goal
def navigate_to_goal(goal):
    distance = np.linalg.norm(goal)
    theta = np.arctan2(goal[1], goal[0])
    # Calculate the throttle and steering values based on the distance
    # 油门，转向，刹车最大值小于1
    throttle = 0.4 # Adjust the constant value as needed
    if theta > 0:
        steering = min(theta, 0.5)  # Adjust the constant value as needed
    else:
        steering = max(theta, -0.5)
    brake = 0
    # velocity = [client.simGetVehiclePose().orientation.x_val,
    #     client.simGetVehiclePose().orientation.y_val,
    #     client.simGetVehiclePose().orientation.z_val]
    # velocity_norm = np.linalg.norm(velocity)
    if distance < 5 :
        brake = 1
    # if velocity_norm > 0.01 and distance < 5:
    #     brake = 1000
    client.setCarControls(airsim.CarControls(throttle, steering, brake))

def next_goal():
    cylinders = client.simGetDetections(camera_name, image_type)
    if not cylinders:
        return None

    min_distance = float('inf')
    closest_cylinder_to_goal = None

    for cylinder in cylinders:
        cylinder_position = np.array([
            cylinder.relative_pose.position.x_val,
            cylinder.relative_pose.position.y_val,
            cylinder.relative_pose.position.z_val
        ])
        distance = np.linalg.norm(cylinder_position - destination_goal)
        
        if distance < min_distance:
            min_distance = distance
            closest_cylinder_to_goal = cylinder
    if closest_cylinder_to_goal:
        return np.array([
            closest_cylinder_to_goal.relative_pose.position.x_val,
            closest_cylinder_to_goal.relative_pose.position.y_val,
            closest_cylinder_to_goal.relative_pose.position.z_val
        ])
    else:
        return None

def detect_obstacle():
    distance_data = client.getDistanceSensorData(distance_sensor_name='DistanseSensor1',vehicle_name='Car1')
    if distance_data.distance < 5:  # If obstacle is closer than 5 meters
        print("Obstacle detected! Turn.")
        return True
    return False

def navigate_to_destination(current_position, distance):
    theta_destination = np.arctan2(destination_goal[1], destination_goal[0])
    theta_current_position = np.arctan2(current_position.y_val, current_position.x_val)
    theta = theta_destination - theta_current_position
    # Calculate the throttle and steering values based on the distance
    # 油门，转向，刹车最大值小于1
    throttle = 0.4 # Adjust the constant value as needed
    if theta > 0:
        steering = min(theta, 0.5)  # Adjust the constant value as needed
    else:
        steering = max(theta, -0.5)
    brake = 0
    if distance < 8:
        brake = 1
    client.setCarControls(airsim.CarControls(throttle, steering, brake))

def reach_to_goal():
    while True:
        current_position = client.simGetVehiclePose().position
        print(current_position)
        distance = np.linalg.norm([
            current_position.x_val- destination_goal[0],
            current_position.y_val - destination_goal[1],
            current_position.z_val - destination_goal[2]
        ])
        navigate_to_destination(current_position, distance)


while True:
    rawImage = client.simGetImage(camera_name, image_type)
    if not rawImage:
        continue
    png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
    current_position = client.simGetVehiclePose().position
    current_theta = np.arctan2(current_position.y_val-destination_goal[1], current_position.x_val-destination_goal[0])
    start_time = time.time()

    # if current_theta - theta_to_goal > np.pi/4:
    #     while True:
    #         client.setCarControls(airsim.CarControls(throttle=1, steering=-0.5, brake=0))
    #         if time.time() - start_time >= 0.1:
    #             break
    # elif current_theta - theta_to_goal < -np.pi/4:
    #     while True:
    #         client.setCarControls(airsim.CarControls(throttle=1, steering=0.5, brake=0))
    #         if time.time() - start_time >= 0.1:
    #             break

    cylinders = client.simGetDetections(camera_name, image_type)
    if cylinders:
        min_cylinder = None
        min_dis = float('inf')
        for cylinder in cylinders:
            current_obj_position = client.simGetObjectPose(cylinder.name).position
            dis_to_dest = np.linalg.norm([
                current_obj_position.x_val - destination_goal[0],
                current_obj_position.y_val - destination_goal[1],
                current_obj_position.z_val - destination_goal[2]
            ])
            dis_to_cylinder = np.linalg.norm([
                current_obj_position.x_val - current_position.x_val,
                current_obj_position.y_val - current_position.y_val,
                current_obj_position.z_val - current_position.z_val
            ])
            if dis_to_dest < min_dis and dis_to_cylinder < 40:
                min_dis = dis_to_dest
                min_cylinder = cylinder

        if min_cylinder:
            min_cylinder_info = [
                min_cylinder.name,
                min_cylinder.relative_pose.position.x_val,
                min_cylinder.relative_pose.position.y_val,
                min_cylinder.relative_pose.position.z_val
            ]
            
            # Drawing rectangle and text on the image
            cv2.rectangle(
                png, 
                (int(min_cylinder.box2D.min.x_val), int(min_cylinder.box2D.min.y_val)), 
                (int(min_cylinder.box2D.max.x_val), int(min_cylinder.box2D.max.y_val)), 
                (255, 0, 0), 
                2
            )
            cv2.putText(
                png, 
                min_cylinder.name, 
                (int(min_cylinder.box2D.min.x_val), int(min_cylinder.box2D.min.y_val) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (36, 255, 12)
            )

        # Optional: Print the min cylinder details
        if min_cylinder:
            # Navigate to the min cylinder position
            target_position = np.array(
                [min_cylinder.relative_pose.position.x_val,
                min_cylinder.relative_pose.position.y_val,
                min_cylinder.relative_pose.position.z_val]
            )
            print(min_cylinder_info, 'Navigating to:', target_position)
            if detect_obstacle():
                start_time = time.time()
                while True:
                    theta = np.arctan2(destination_goal[1]-current_position.y_val, 
                                    destination_goal[0]- current_position.x_val)
                    client.setCarControls(airsim.CarControls(throttle=1, steering=-min(theta,0.5), brake=0))
                    if time.time() - start_time >= 1.0:
                        break
            else:
                navigate_to_goal(target_position)

            # current_position = client.simGetVehiclePose().position
            # if np.linalg.norm([target_position[0],target_position[1],target_position[2]]) < radius:
            #     next_target = next_goal()
            #     if next_target is not None:
            #         navigate_to_goal(next_target)
            #     else:
            #         print("No more cylinders detected, stopping navigation.")
            #         break

        else:
            client.setCarControls(airsim.CarControls(throttle=1, steering=0, brake=0.5))

    # Check if we reached the final destination
    current_position = client.simGetVehiclePose().position
    if np.linalg.norm([
                current_position.x_val- destination_goal[0],
                current_position.y_val - destination_goal[1],
                current_position.z_val - destination_goal[2]
            ]) < radius:
        print("Reaching to destination goal!")
        reach_to_goal()
        break

    cv2.imshow("AirSim", png)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('c'):
        client.simClearDetectionMeshNames(camera_name, image_type)
    elif cv2.waitKey(1) & 0xFF == ord('a'):
        client.simAddDetectionFilterMeshName(camera_name, image_type, "Cylinder*")

cv2.destroyAllWindows()
