import argparse
from pathlib import Path
import time

import open3d
from mayavi import mlab

from vgn.constants import vgn_res
from vgn.grasp_detector import GraspDetector
from vgn.perception.integration import TSDFVolume
from vgn.utils import vis
from vgn.utils.data import SceneData
from vgn.utils.io import load_dict


def main(args):
    config = load_dict(Path(args.config))
    vol_size = config["vol_size"]
    vol_res = vgn_res
    threshold = config["threshold"]

    # Load scene data
    scene_dir = Path(args.scene)
    scene = SceneData.load(scene_dir)

    # Build TSDF
    tsdf = TSDFVolume(vol_size, vol_res)
    tsdf.integrate_images(scene.depth_imgs, scene.intrinsic, scene.extrinsics)
    point_cloud = tsdf.extract_point_cloud()
    tsdf_vol = tsdf.get_volume()

    # Detect grasps
    detector = GraspDetector(Path(args.model), vol_size)
    tic = time.time()
    grasps, qualities, info = detector.detect_grasps(tsdf_vol, 0.8)
    toc = time.time() - tic
    print("Prediction took {} s".format(toc))

    # Plot TSDF voxel volume
    mlab.figure("TSDF volume")
    vis.draw_volume(tsdf_vol, tsdf.voxel_size)

    # Draw network output, overlaid with point cloud and detected grasps
    mlab.figure("Grasp quality volume")
    vis.draw_volume(info["filtered"], tsdf.voxel_size)
    vis.draw_point_cloud(point_cloud)
    vis.draw_grasps(grasps, qualities, draw_frames=True)

    mlab.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="detect grasps in a scene")
    parser.add_argument(
        "--model", type=str, required=True, help="saved model ending with .pth"
    )
    parser.add_argument("--scene", type=str, required=True, help="scene directory")
    parser.add_argument(
        "--config",
        type=str,
        default="config/grasp_detection.yaml",
        help="path to grasp detection configuration",
    )
    args = parser.parse_args()
    main(args)
