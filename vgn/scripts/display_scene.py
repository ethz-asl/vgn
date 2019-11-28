import argparse
from pathlib import Path

from vgn.utils.io import load_dict

from vgn.utils.vis import display_scene
from vgn.utils.data import SceneData


def main(args):
    scene_dir = Path(args.scene)
    assert scene_dir.exists(), "The given scene does not exist"

    data_gen_config = load_dict(Path(args.data_gen_config))
    scene_data = SceneData.load(scene_dir)

    display_scene(
        scene_data=scene_data,
        vol_size=data_gen_config["vol_size"],
        vol_res=data_gen_config["vol_res"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize data from a scene")
    parser.add_argument("--scene", type=str, required=True, help="scene directory")
    parser.add_argument(
        "--data-gen-config",
        type=str,
        default="config/data_generation.yaml",
        help="path to data generation configuration",
    )
    args = parser.parse_args()
    main(args)
