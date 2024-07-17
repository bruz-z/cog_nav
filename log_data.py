import airsim
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
CLIENT = airsim.CarClient()
CLIENT.confirmConnection()
CLIENT.enableApiControl(False)

with open('vehicle_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'LinearVelocityX', 'LinearVelocityY', 
                     'AngularVelocityZ', 
                     'PositionX', 'PositionY', 'PositionZ',
                     'OrientationX', 'OrientationY', 'OrientationZ', 'OrientationW'])

    while True:
        car_state = CLIENT.getCarState()
        linear_velocity = car_state.kinematics_estimated.linear_velocity
        angular_velocity = car_state.kinematics_estimated.angular_velocity 
        position = car_state.kinematics_estimated.position
        orientation = car_state.kinematics_estimated.orientation
        timestamp = time.time()

        # 写入CSV文件
        writer.writerow([timestamp, linear_velocity.x_val, linear_velocity.y_val, 
                         angular_velocity.z_val,
                         position.x_val, position.y_val, position.z_val,
                         orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val])
        
        # 控制记录频率
        time.sleep(0.01)
