#%%
import numpy as np
from pathlib import Path
import pandas as pd

#%% Read
root = Path("../data/grasps/blocks")
df = pd.read_csv(root / "grasps.csv")

#%% Write
df.to_csv(root / "grasps.csv", index=False)

#%%
pos = df[df.physics == 1]
neg = df[df.physics == 0]

print("number of scenes:", len(np.unique(df.scene_id)))
print("Number of grasps:", len(df.index))
print("Number of positives:", len(pos.index))
print("Number of negatives:", len(neg.index))

#%% Remove grasp positions that lie outside the workspace
df.drop(
    df[
        (df.x < 0.04)
        | (df.x > 0.26)
        | (df.y < 0.04)
        | (df.y > 0.26)
        | (df.z < 0.04)
        | (df.z > 0.26)
    ].index,
    inplace=True,
)

#%% Balance
pos = df[df.physics == 1]
neg = df[df.physics == 0]
pos_count = len(pos.index)
neg_count = len(neg.index)
i = np.random.choice(neg.index, neg_count - 2 * pos_count, replace=False)
df = df.drop(i)

#%% Delete unreferenced scenes
scenes = df.scene_id.values
for f in root.iterdir():
    if f.suffix == ".npz" and f.stem not in scenes:
        print("Removed", f)
        f.unlink()

# %% Merge folders
target_foler = "../data/grasps/blocks"
source_folders = ["../data/grasps/blocks_euler", "../data/grasps/blocks_osaka"]
df = pd.concat([pd.read_csv(root + "/grasps.csv") for root in source_folders])
df.to_csv(target_foler + "/grasps.csv", index=False)

# %%
