import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None, posn_tolerance=0.1, ang_pos_tolerance=0.1, 
                 target_vel=None, vel_tolerance=0.01, target_ang_vel=None, ang_vel_tolerance=0.01):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
            pos_tolerance: tolerance on the position in all of x, y and z
            ang_pos_tolerance: tolerance on the Euler angles
            target_vel: target/goal velocities in x, y, z for the agent
            vel_tolerance: tolerance on the velocity in all of x, y and z
            target_ang_vel: target angular velocities for the three Euler angles
            ang_vel_tolerance: tolerance on the angular velocity in all 3 of the Eular angles
        """
        # Simulation
        self.sim_time = 0.0 # Not amending sim but would like time in the reward hence hard coding these values
        self.sim_dt = 1 / 50.0
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        self.target_vel = target_vel if target_vel is not None else np.array([0., 0., 0.])
        self.target_ang_vel = target_ang_vel if target_ang_vel is not None else np.array([0., 0., 0.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # Starter reward...
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        # Goal is to hover at start position
        # be close to start position
        distance_from_goal = np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        
        # be not moving up/down/left/right/forward/backwards
        speed = abs(self.sim.v).sum()
        
        # be not spinning
        ang_speed = abs(self.sim.angular_v).sum()
        
        if distance_from_goal <= pos_tolerance: # reward for being within tolerance near the goal
            if speed <= vel_tolerance and ang_speed <= ang_vel_tolerance: # reward for being broadly stationary
                # quad is near goal and broadly stationary
                reward = 10.0
            else:
                # reduce reward proportional to speed/ang_speed
                reward = 10.0 - 0.2 * speed - 0.2 * ang_speed
        else: # the quad is not near enough to the goal therefore it should be moving
            # reward based solely on distance at this point
            reward = -1.0 * distance
            
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0.0
        pose_all = []
        for _ in range(self.action_repeat):
            # update the time like in the sim...
            self.sim_time += self.sim_dt
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state