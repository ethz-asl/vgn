class Grasp(object):
    """Grasp parameterized as pose of a 2-finger robot hand.
    
    TODO(mbreyer): clarify definition of grasp frame
    """

    def __init__(self, pose, width):
        self.pose = pose
        self.width = width
