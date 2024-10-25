#Importing the necessary modules
from controller import Robot

#Defining the run_robot function
def run_robot(robot):
    timestep = int(robot.getBasicTimeStep()) #time step and maximum speed
    max_speed = 6.28
    
     #Initializing left and right motors:
    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    
   
   #starting motor positions and velocities
    left_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setPosition(float('inf'))
    right_motor.setVelocity(0.0)
    
    #Initializing prox sensors
    prox_sensors = []
    for ind in range(8):
        sensor_name = 'ps' + str(ind)
        prox_sensors.append(robot.getDevice(sensor_name)) 
        prox_sensors[ind].enable(timestep)
    
    # Enable the camera
    camera = robot.getDevice("camera")
    camera.enable(timestep)
     
     
    #main loop for control   
    while robot.step(timestep) != -1:
            
        left_wall = prox_sensors[5].getValue() > 80
        left_corner = prox_sensors[6].getValue() > 80
        front_wall = prox_sensors[7].getValue() > 80
        
        # Get image from camera
        image = camera.getImage()
        
        left_speed = max_speed
        right_speed = max_speed
        
        #motor speeds based on sensor values
        if front_wall:
            print("Turn right in place")
            left_speed = max_speed
            right_speed = -max_speed
        else:
            if left_wall:
                print("Drive forward")
                left_speed = max_speed
                right_speed = max_speed
            else:
                print("Turn left")
                left_speed = max_speed / 8
                right_speed = max_speed
                
            if left_corner:
                print("Came too close, drive right")
                left_speed = max_speed
                right_speed = max_speed / 8
        
        #motor velocity        
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)
        
if __name__ == "__main__":
    my_robot = Robot()
    run_robot(my_robot)
