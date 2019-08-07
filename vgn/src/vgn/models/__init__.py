def get_model(name):
    name = name.lower()
    if name == 'voxnet':
        from vgn.models.voxnet import VoxNet
        return VoxNet()
