# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Hans Baumann-Ortiz
# Licensed under the Apache License 2.0


from isaaclab.utils import configclass


#from unitree_rl_lab.robots.anymal import ANYMAL_D_CFG
#import unitree assets and motors
from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG




from isaaclab.assets import ArticulationCfg
from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
import torch




# Anymal joint values
# ============================================================
# Joint Name                | Lower Limit  | Upper Limit 
# ------------------------------------------------------------
# LF_HAA                    |      -0.7505 |       0.5760
# LF_HFE                    |      -8.9535 |       8.9535
# LF_KFE                    |      -8.9535 |       8.9535
# RF_HAA                    |      -0.5760 |       0.7505
# RF_HFE                    |      -8.9535 |       8.9535
# RF_KFE                    |      -8.9535 |       8.9535
# LH_HAA                    |      -0.7505 |       0.5760
# LH_HFE                    |      -8.9535 |       8.9535
# LH_KFE                    |      -8.9535 |       8.9535
# RH_HAA                    |      -0.5760 |       0.7505
# RH_HFE                    |      -8.9535 |       8.9535
# RH_KFE                    |      -8.9535 |       8.9535
# ============================================================


# ------------------
#unitree motor condif values:


   # Unitree actuator class that implements a torque-speed curve for the actuators.


   #     The torque-speed curve is defined as follows:


   #             Torque Limit, N·m
   #                 ^
   #     Y2──────────|
   #                 |──────────────Y1
   #                 |              │\
   #                 |              │ \
   #                 |              │  \
   #                 |              |   \
   #     ------------+--------------|------> velocity: rad/s
   #                               X1   X2


   #     - Y1: Peak Torque Test (Torque and Speed in the Same Direction)
   #     - Y2: Peak Torque Test (Torque and Speed in the Opposite Direction)
   #     - X1: Maximum Speed at Full Torque (T-N Curve Knee Point)
   #     - X2: No-Load Speed Test


   #     - Fs: Static friction coefficient
   #     - Fd: Dynamic friction coefficient
   #     - Va: Velocity at which the friction is fully activated


# The GO2 uses this motor config:


   # @configclass
   # class UnitreeActuatorCfg_Go2HV(UnitreeActuatorCfg):
   #     X1 = 13.5
   #     X2 = 30
   #     Y1 = 20.2
   #     Y2 = 23.4








# UNITREE_GO2_CFG = UnitreeArticulationCfg(
#     spawn=UnitreeUrdfFileCfg(
#         asset_path=f"{UNITREE_ROS_DIR}/robots/go2_description/urdf/go2_description.urdf",
#     ),
#     # spawn=UnitreeUsdFileCfg(
#     #     usd_path=f"{UNITREE_MODEL_DIR}/Go2/usd/go2.usd",
#     # ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.4),
#         joint_pos={
#             ".*R_hip_joint": -0.1,
#             ".*L_hip_joint": 0.1,
#             "F[L,R]_thigh_joint": 0.8,
#             "R[L,R]_thigh_joint": 1.0,
#             ".*_calf_joint": -1.5,
#         },
#         joint_vel={".*": 0.0},
#     ),
#     actuators={
#         "GO2HV": unitree_actuators.UnitreeActuatorCfg_Go2HV(
#             joint_names_expr=[".*"],
#             stiffness=25.0,
#             damping=0.5,
#             friction=0.01,
#         ),
#     },
#     # fmt: off
#     joint_sdk_names=[
#         "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
#         "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
#         "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
#         "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
#     ],
#     # fmt: on
# )
# -----------------


# ============================================================
# Joint Name                | Lower Limit  | Upper Limit 
# ------------------------------------------------------------
# FR_hip_joint              |      -0.9425 |       0.9425
# FR_thigh_joint            |      -1.3177 |       3.2376
# FR_calf_joint             |      -2.6285 |      -0.9320
# FL_hip_joint              |      -0.9425 |       0.9425
# FL_thigh_joint            |      -1.3177 |       3.2376
# FL_calf_joint             |      -2.6285 |      -0.9320
# RR_hip_joint              |      -0.9425 |       0.9425
# RR_thigh_joint            |      -0.2705 |       4.2848
# RR_calf_joint             |      -2.6285 |      -0.9320
# RL_hip_joint              |      -0.9425 |       0.9425
# RL_thigh_joint            |      -0.2705 |       4.2848
# RL_calf_joint             |      -2.6285 |      -0.9320
# ============================================================


# ============================================================

GO2HV_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
   joint_names_expr=[".*"],
   stiffness={".*": 25.0},  # P gain in Nm/rad
   damping={".*": 0.5},  # D gain in Nm s/rad
   saturation_effort=23.4, # Unitree Y2
   effort_limit=20.2,      # Unitree Y1
   velocity_limit=30.0,    # Unitree X2
   encoder_bias=[0.0] * 12,  # encoder bias in radians
   max_delay=10,  # max delay in simulation steps
   #friction=0.01, not in the example cfg for pace
)


@configclass
class GO2PaceCfg(PaceCfg):
   """Pace configuration for GO2 robot."""
   robot_name: str = "go2_sim"
   data_dir: str = "go2_sim/chirp_data.pt"  # located in pace_sim2real/data/anymal_d_sim/chirp_data.pt
   bounds_params: torch.Tensor = torch.zeros((49, 2))  # 12 + 12 + 12 + 12 + 1 = 49 parameters to optimize
   joint_order: list[str] = [
       # same order as in unitree.py
       "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
       "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
       "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
       "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
   ]


   def __post_init__(self):
       # set bounds for parameters
       #for go2 damping smaller, smaller max armature
       self.bounds_params[:12, 0] = 1e-5
       self.bounds_params[:12, 1] = 0.05  # armature between 1e-5 - 1.0 [kgm2]
       self.bounds_params[12:24, 1] = 3.0  # dof_damping between 0.0 - 7.0 [Nm s/rad]
       self.bounds_params[24:36, 1] = 0.5  # friction between 0.0 - 0.5
       self.bounds_params[36:48, 0] = -0.1
       self.bounds_params[36:48, 1] = 0.1  # bias between -0.1 - 0.1 [rad]
       self.bounds_params[48, 1] = 10.0  # delay between 0.0 - 10.0 [sim steps]




@configclass
class GO2PaceSceneCfg(PaceSim2realSceneCfg):
   """Configuration for GO2 robot in Pace Sim2Real environment."""
   robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
       prim_path="{ENV_REGEX_NS}/Robot",
       init_state=ArticulationCfg.InitialStateCfg(
           pos=(0.0, 0.0, 1.0),
           joint_pos={
            ".*R_hip_joint": 0.0,
            ".*L_hip_joint": 0.0,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
               },
       ),
       actuators={"GO2HV": GO2HV_PACE_ACTUATOR_CFG})




@configclass
class GO2PaceEnvCfg(PaceSim2realEnvCfg):


   scene: GO2PaceSceneCfg = GO2PaceSceneCfg()
   sim2real: PaceCfg = GO2PaceCfg()


   def __post_init__(self):
       # post init of parent
       super().__post_init__()


       # robot sim and control settings
       self.sim.dt = 0.0025  # 400Hz simulation
       self.decimation = 1  # 400Hz control





