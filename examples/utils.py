import genesis as gs
import numpy as np
import torch
from pytransform3d import (
    transformations as pt,
    rotations as pr,
    batch_rotations as pb,
    trajectories as ptr,
    plot_utils as ppu
)
from pandaSim.geometry.genesis_adapter import GenesisAdapter
from pandaSim.planning.screw_motion_planner import ScrewMotionPlanner

import spatialmath as sm

# Define joint groups
motors_dof = np.arange(7)  # Joints 0-6 for arm movement
fingers_dof = np.arange(7, 9)  # Joints 7-8 for gripper fingers
virtual_finger_idx = 9  # Joint 9 for virtual finger
spinning_pads = np.arange(10, 12)  # Joints 10-11 for spinning pads

class RobotController:
    """A class to encapsulate robot control and simulation dependencies"""
    
    def __init__(self):
        """Initialize the simulation environment and robot controller"""
        gs.destroy()
        gs.init(backend=gs.gpu)
        gs.set_random_seed(seed=42)
        self.adapter = GenesisAdapter()
        self.planner = ScrewMotionPlanner()
        self.scene = self.adapter.scene
        self.franka = None
        self.target_entity = None
        self.objects_list = []
        self.scene_list = []
        
    def init_scene(self, record=False):
        """Initialize the scene with the objects and the robot"""
        plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda_vir.xml")
        )

        # Create tables
        table1 = self.scene.add_entity(
            gs.morphs.Box(
                size = (0.7, 1.7, 0.02),
                pos = (0.5, 0, 0.19),
                fixed = True,
            ),
            surface=gs.surfaces.Default(
                color = (0.5, 0.5, 0.5),
            ),
            material=gs.materials.Rigid(
                gravity_compensation=1.0,
                friction=5.0
            )
        )
                
        table2 = self.scene.add_entity(
            gs.morphs.Box(
                size = (0.7, 1.7, 0.02),
                pos = (-0.5, 0, 0.19),
                fixed = True,
            ),
            surface=gs.surfaces.Default(
                color = (0.5, 0.5, 0.5),
            ),
            material=gs.materials.Rigid(
                gravity_compensation=1.0,
                friction=5.0
            )
        )

        table3 = self.scene.add_entity(
            gs.morphs.Box(
                size = (0.3, 0.7, 0.02),
                pos = (0, -0.5, 0.19),
                fixed = True,
            ),
            surface=gs.surfaces.Default(
                color = (0.5, 0.5, 0.5),
            ),
            material=gs.materials.Rigid(
                gravity_compensation=1.0,
                friction=5.0
            )
        )

        table4 = self.scene.add_entity(
            gs.morphs.Box(
                size = (0.3, 0.7, 0.02),
                pos = (0, 0.5, 0.19),
                fixed = True,
            ),
            surface=gs.surfaces.Default(
                color = (0.5, 0.5, 0.5),
            ),
            material=gs.materials.Rigid(
                gravity_compensation=1.0,
                friction=5.0
            )
        )

        # Create objects
        cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.1, 0.07, 0.25),
                pos=(0.4, 0.1, 0.05+0.2),
                euler=(45, -90, 0)
            ),
            surface=gs.surfaces.Default(
                color=(0.5, 0.8, 0.94),
            ),
            material=gs.materials.Rigid(rho = 300,friction=5)
        )

        bottle = self.scene.add_entity(
            material=gs.materials.Rigid(rho=300),
            morph=gs.morphs.URDF(
                file="urdf/3763/mobility_vhacd.urdf",
                scale=0.09,
                pos=(0.3, -0.3, 0.036+0.2),
                euler=(-45, -90, 0),
            ),
        )

        cylinder = self.scene.add_entity(
            morph=gs.morphs.Cylinder(
                radius=0.03,
                height=0.15,
                pos=(0.3, 0.3, 0.03+0.2),
                euler=(90, -90, 0),
            ),
            surface=gs.surfaces.Default(
                color=(0.3, 0.47, 0.17),
            ),
            material=gs.materials.Rigid(rho = 300,friction=5)
        )

        # Create target entity
        self.target_entity = self.scene.add_entity(
            gs.morphs.Mesh(
                file="meshes/axis.obj",
                scale=0.15,
                collision=False,
            ),
            surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )

        if record:
            self.scene.cam = self.scene.add_camera(
                res=(1280, 960),
                pos=(2.0, 1.0, 1.5),
                lookat=(0, 0, 0.5),
                fov=40,
                GUI=False,
            )
        else:
            self.scene.cam = None
        


        cube.name = "cube"
        bottle.name = "bottle"
        cylinder.name = "cylinder"

        self.objects_list = [cylinder,cube, bottle]
        self.scene_list = [plane, table1, table2, table3, table4]

        
        return self.franka, self.objects_list, self.target_entity, self.scene_list, self.scene.cam

    def move_to_position(self, target_link, position, orientation, fingers_state=0.04, num_waypoints=150):
        """Move the robot to a specific position and orientation"""
        qpos = self.franka.inverse_kinematics(
            link=target_link,
            pos=position,
            quat=orientation,
            pos_tol=1e-5,
            rot_tol=1e-5,
            max_solver_iters=100,
        )
        qpos[fingers_dof] = fingers_state
        
        path = self.franka.plan_path(
            qpos_goal=qpos,
            num_waypoints=num_waypoints
        )
        
        for waypoint in path:
            self.franka.control_dofs_position(waypoint)
            self.scene.step()
            if self.scene.cam:
                self.scene.cam.render()
        
        self.adapter.step_simulation(0.3)
        return qpos

    def compute_object_grasp(self, obj, grasp_height='center', 
                           offset_toward=0.04, offset_upward=0.02,
                           grasp_T_gripper=None, prefer_closer_grasp=True):
        """Compute a grasp pose for the given object"""
        self.scene.step()
        
        if grasp_T_gripper is None:
            grasp_T_gripper = np.array([
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]
            ])
        
        grasp_pose, qs, s_axes = self.planner.compute_grasp(
            obj=obj,
            adapter=self.adapter,
            prefer_closer_grasp=prefer_closer_grasp,
            grasp_height=grasp_height,
            offset_toward=offset_toward,
            offset_upward=offset_upward,
            gripper_offset=grasp_T_gripper,
            output_type='pq'
        )
        
        self.target_entity.set_qpos(grasp_pose)
        self.scene.step()
        s_axes[..., 2] = 0
        lower, upper = obj.get_AABB().cpu().numpy()
        qs[..., 2] = lower[..., 2]
        return grasp_pose, qs, s_axes

    def pregrasp_object(self, finger_link, grasp_pose, fingers_state=0.04):
        pregrasp_pos = grasp_pose[:3] + np.array([0, 0, 0.1])
        self.move_to_position(finger_link, pregrasp_pos, grasp_pose[3:], fingers_state=fingers_state, num_waypoints=50)


    def grasp_object(self, finger_link, grasp_pose, fingers_state=0.04, fingers_force = 30):     
        self.move_to_position(finger_link, grasp_pose[:3], grasp_pose[3:], fingers_state=fingers_state, num_waypoints=50)
        
        self.franka.control_dofs_force(np.array([-fingers_force, -fingers_force]), fingers_dof)
        self.adapter.step_simulation(0.5)

    def object_trajectory(self, finger_link, obj, qs, s_axes):
        obj_initial_pose = self.adapter.get_pose(obj, 't')
        trajectory = self.planner.plan(
            robot=self.franka,
            link=finger_link,
            object=obj,
            adapter=self.adapter,
            initial_pose=obj_initial_pose,
            qs=qs,
            s_axes=s_axes,
            theta=np.pi/2,
            output_type='t'
        )
        return trajectory
    
    def execute_manipulation(self, finger_link, obj, qs, s_axes):
        """Execute planned manipulation with the grasped object"""
        trajectory = self.planner.plan(
            robot=self.franka,
            link=finger_link,
            object=obj,
            adapter=self.adapter,
            qs=qs,
            s_axes=s_axes,
            theta=np.pi/2 + 0.03,
            output_type='pq'
        )
        
        qposs = []
        qpos = self.franka.get_qpos()
        for pq in trajectory:
            qpos = self.franka.inverse_kinematics(
                link=finger_link,
                pos=pq[:3],
                quat=pq[3:],
                init_qpos=qpos,
                pos_tol=1e-6,
                rot_tol=1e-6,
                max_solver_iters=100,
            )
            qposs.append(qpos)
        
        for waypoint in qposs:
            self.franka.control_dofs_position(waypoint[motors_dof], motors_dof)
            self.target_entity.set_pos(finger_link.get_pos())
            self.target_entity.set_quat(finger_link.get_quat())
            self.scene.step()
            if self.scene.cam:
                self.scene.cam.render()

    def release_object(self):
        """Open the gripper to release the object"""
        self.franka.control_dofs_force(np.array([0, 0]), fingers_dof)
        self.franka.control_dofs_position(np.array([0.04, 0.04]), fingers_dof)
        self.adapter.step_simulation(0.1)

    def move_up_from_current(self, finger_link, height=0.1):
        """Move up by specified height from current position"""
        current_pose = self.adapter.forward_kinematics(self.franka, finger_link, output_type='pq')
        target_pos = current_pose[:3] + np.array([0, 0, height])
        self.move_to_position(finger_link, target_pos, current_pose[3:], num_waypoints=50)

    def return_to_ready_pose(self):
        """Return the robot to its ready pose"""
        path = self.franka.plan_path(
            qpos_goal=self.franka.ready_qpos,
            num_waypoints=100
        )
        
        for waypoint in path:
            self.franka.control_dofs_position(waypoint)
            self.scene.step()
            if self.scene.cam:
                self.scene.cam.render()
        
        self.adapter.step_simulation(0.1)
    
    def Success_Rate(self, initial_pose, goal_pose, current_pose, method):
        """
        Compare the goal and current transformation and give a success rate based on that.
        """
        
        threshold = 0.2
        # Compute pose difference in SE(3)
        eTep = np.linalg.inv(current_pose) @ goal_pose
        
        # Initialize error vector
        e_log = np.zeros(6)
        e_rpy = np.zeros(6)
        e = np.zeros(6)
        
        if method.lower().startswith("a"):
            # Get translational error
            e_log[:3] = eTep[:3, 3]
            
            # Get rotational error as axis-angle
            rot_matrix = eTep[:3, :3]
            axis_ang = pr.axis_angle_from_matrix(rot_matrix)
            axis = axis_ang[:3]
            angle = axis_ang[3]
            e_log[3:] = axis * angle if angle != 0 else np.zeros(3)
            # Check if arrived at target
            arrived = np.linalg.norm(np.abs(e_log)) < threshold
            e = e_log
        elif method.lower().startswith("r"):
            # RPY method
            e_rpy[:3] = eTep[:3, 3]
            e_rpy[3:] = pr.euler_from_matrix(eTep[:3, :3], 0, 1, 2, extrinsic=False)
            # Check if arrived at target
            arrived = np.linalg.norm(np.abs(e_rpy)) < threshold
            e = e_rpy
        #elif method =='log':
            
        angular_rate = 1 - np.linalg.norm(np.abs(e[3:])) / (np.pi/2)   
        trans_rate = 1 - np.linalg.norm(np.abs(e[:3])) / np.linalg.norm(np.abs(goal_pose[:3,3]- initial_pose[:3,3]))
        Success = ((angular_rate+trans_rate)/2)*100
        return e, Success, arrived