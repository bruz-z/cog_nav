import airsim
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from gridcell import plot_gridcell

# Connect to the AirSim simulator
CLIENT = airsim.CarClient()
CLIENT.confirmConnection()

# Set camera name and image type to request images and detections
CAMERA_NAME = "0"
IMAGE_TYPE = airsim.ImageType.Scene
CLIENT.enableApiControl(True)
# Set detection RADIUS in [cm]
CLIENT.simSetDetectionFilterRadius(CAMERA_NAME, IMAGE_TYPE, 600 * 300)
# Add desired object name to detect in wild card/regex format
CLIENT.simAddDetectionFilterMeshName(CAMERA_NAME, IMAGE_TYPE, "Cylinder*")

# Set the grid cell parameters
ENV_SIZE = 50
WAVELENGTH = 20
DIRECTION = 0
INITIAL_POSITION = (0, 0)

# Set the destination goal
DESTINATION_GOAL = np.array([ENV_SIZE, ENV_SIZE, 0])
THETA_TO_GOAL = np.arctan2(DESTINATION_GOAL[1], DESTINATION_GOAL[0])
RADIUS = 15

# Function to navigate to a given goal
def navigate_to_goal(goal):
    distance = np.linalg.norm(goal)
    theta = np.arctan2(goal[1], goal[0])
    # Calculate the throttle and steering values based on the distance and grid cell firing pattern
    throttle = 0.4  # Adjust the constant value as needed
    if theta > 0:
        steering = min(theta, 0.5)  # Adjust the constant value as needed
    else:
        steering = max(theta, -0.5)
    brake = 0
    if distance < 5:
        brake = 1
    CLIENT.setCarControls(airsim.CarControls(throttle, steering, brake))

def next_goal():
    cylinders = CLIENT.simGetDetections(CAMERA_NAME, IMAGE_TYPE)
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
        distance = np.linalg.norm(cylinder_position - DESTINATION_GOAL)

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
    distance_data = CLIENT.getDistanceSensorData(distance_sensor_name='DistanseSensor1', vehicle_name='Car1')
    if distance_data.distance < 5:  # If obstacle is closer than 5 meters
        print("Obstacle detected! Turn.")
        return True
    return False

def navigate_to_destination(current_position, distance):
    theta_destination = np.arctan2(DESTINATION_GOAL[1], DESTINATION_GOAL[0])
    theta_current_position = np.arctan2(current_position.y_val, current_position.x_val)
    theta = theta_destination - theta_current_position
    # Calculate the throttle and steering values based on the distance and grid cell firing pattern
    throttle = 0.4  # Adjust the constant value as needed
    if theta > 0:
        steering = min(theta, 0.5)  # Adjust the constant value as needed
    else:
        steering = max(theta, -0.5)
    brake = 0
    if distance < 8:
        brake = 1
    CLIENT.setCarControls(airsim.CarControls(throttle, steering, brake))

def reach_to_goal():
    while True:
        current_position = CLIENT.simGetVehiclePose().position
        print(current_position)
        distance = np.linalg.norm([
            current_position.x_val - DESTINATION_GOAL[0],
            current_position.y_val - DESTINATION_GOAL[1],
            current_position.z_val - DESTINATION_GOAL[2]
        ])
        navigate_to_destination(current_position, distance)

plot_gridcell(ENV_SIZE, WAVELENGTH, DIRECTION, INITIAL_POSITION)

while True:
    rawImage = CLIENT.simGetImage(CAMERA_NAME, IMAGE_TYPE)
    if not rawImage:
        continue
    png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
    current_position = CLIENT.simGetVehiclePose().position
    current_theta = np.arctan2(current_position.y_val - DESTINATION_GOAL[1], current_position.x_val - DESTINATION_GOAL[0])
    start_time = time.time()
    cylinders = CLIENT.simGetDetections(CAMERA_NAME, IMAGE_TYPE)

    imu_data = CLIENT.getImuData()
    print(imu_data)
    if cylinders:
        min_cylinder = None
        min_dis = float('inf')
        for cylinder in cylinders:
            current_obj_position = CLIENT.simGetObjectPose(cylinder.name).position
            dis_to_dest = np.linalg.norm([
                current_obj_position.x_val - DESTINATION_GOAL[0],
                current_obj_position.y_val - DESTINATION_GOAL[1],
                current_obj_position.z_val - DESTINATION_GOAL[2]
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
                        theta = np.arctan2(DESTINATION_GOAL[1] - current_position.y_val,
                                           DESTINATION_GOAL[0] - current_position.x_val)
                        CLIENT.setCarControls(airsim.CarControls(throttle=1, steering=-min(theta, 0.5), brake=0))
                        if time.time() - start_time >= 1.0:
                            break
                else:
                    navigate_to_goal(target_position)

            else:
                CLIENT.setCarControls(airsim.CarControls(throttle=1, steering=0, brake=0.5))

    # Check if we reached the final destination
    current_position = CLIENT.simGetVehiclePose().position
    if np.linalg.norm([
        current_position.x_val - DESTINATION_GOAL[0],
        current_position.y_val - DESTINATION_GOAL[1],
        current_position.z_val - DESTINATION_GOAL[2]
    ]) < RADIUS:
        print("Reaching to destination goal!")
        reach_to_goal()
        break

    cv2.imshow("AirSim", png)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('c'):
        CLIENT.simClearDetectionMeshNames(CAMERA_NAME, IMAGE_TYPE)
    elif cv2.waitKey(1) & 0xFF == ord('a'):
        CLIENT.simAddDetectionFilterMeshName(CAMERA_NAME, IMAGE_TYPE, "Cylinder*")

cv2.destroyAllWindows()
