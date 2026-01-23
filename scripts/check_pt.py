import torch

def deep_inspect(file_path):
    # Load data to CPU
    data = torch.load(file_path, map_location="cpu")
    t = data["time"]
    target = data["des_dof_pos"]
    
    # Your Exact Joint Names
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
    ]
    
    # Exact Limits from your Table
    limits = {
        "hip": (-0.9425, 0.9425),
        "f_thigh": (-1.3177, 3.2376),
        "r_thigh": (-0.2705, 4.2848),
        "calf": (-2.6285, -0.9320)
    }

    print(f"\n{'JOINT NAME':<20} | {'MIN':>8} | {'MAX':>8} | {'STATUS'}")
    print("-" * 55)

    for i, name in enumerate(joint_names):
        j_min = torch.min(target[:, i]).item()
        j_max = torch.max(target[:, i]).item()
        
        # Determine which limit to check against
        if "hip" in name:
            l_min, l_max = limits["hip"]
        elif "FR_thigh" in name or "FL_thigh" in name:
            l_min, l_max = limits["f_thigh"]
        elif "RR_thigh" in name or "RL_thigh" in name:
            l_min, l_max = limits["r_thigh"]
        else: # calf
            l_min, l_max = limits["calf"]

        # Check for violation
        status = "✅ SAFE"
        if j_min < l_min or j_max > l_max:
            status = "❌ VIOLATION"

        print(f"{name:<20} | {j_min:>8.3f} | {j_max:>8.3f} | {status}")

    # Velocity check
    dt = t[1] - t[0]
    vel = (target[1:] - target[:-1]) / dt
    print(f"\nMax Commanded Velocity: {torch.max(torch.abs(vel)):.2f} rad/s")

if __name__ == "__main__":
    deep_inspect("/home/ubuntu/pace-sim2real/data/go2_sim/chirp_data.pt")