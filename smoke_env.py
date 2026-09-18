import os
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = os.path.join(
    get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
)

env = OffScreenRenderEnv(
    bddl_file_name=bddl,
    camera_heights=128,
    camera_widths=128,
)
env.seed(0)
env.reset()
obs = env.set_init_state(suite.get_task_init_states(0)[0])
obs, reward, done, info = env.step(np.zeros(7))
print(obs["agentview_image"].shape, reward, done)
env.close()

