from tigre.utilities.geometry import Geometry
import numpy as np
import json

# Tigre geometry
class ConeGeometry(Geometry):
    """
    Cone beam CT geometry.
    """

    def __init__(self, data, scale_factor = 1e-2 ):

        Geometry.__init__(self)

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = data["DSD"] * scale_factor  # Distance Source Detector      (m)
        self.DSO = data["DSO"] * scale_factor  # Distance Source Origin        (m)
        # Detector parameters
        self.nDetector = np.array(data["nDetector"])  # number of pixels              (px)
        self.dDetector = np.array(data["dDetector"]) * scale_factor  # size of each pixel            (m)
        self.sDetector = self.nDetector * self.dDetector  # total size of the detector    (m)
        # Image parameters
        self.nVoxel = np.array(data["nVoxel"])[::-1]  # number of voxels              (vx)
        self.dVoxel = np.array(data["dVoxel"]) * scale_factor  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)

        # Offsets
        self.offOrigin = np.array(data["offOrigin"]) * scale_factor  # Offset of image from origin   (m)
        self.offDetector = np.array(
            [data["offDetector"][0], data["offDetector"][1], 0]) * scale_factor  # Offset of Detector            (m)

        # Auxiliary
        self.accuracy = data["accuracy"]  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = data["mode"]  # parallel, cone                ...
        self.filter = data["filter"]

# get near and far threshold
def get_near_far(geo, adjust=0):
    """
    Compute the near and far threshold.; from https://github.com/Ruyi-Zha/naf_cbct/blob/main/src/dataset/tigre.py
    """
    dist1 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
    dist2 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
    dist3 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
    dist4 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
    dist_max = np.max([dist1, dist2, dist3, dist4])
    near = np.max([0, geo.DSO - dist_max - adjust])
    far = np.min([geo.DSO * 2, geo.DSO + dist_max + adjust])

    return near, far

# store general params geometry
def store_general_geo(data_geo, near_thresh, far_thresh, max_pixel_value, train_folder_name, scale_factor=1e-2):
    data_geo['near_thresh'] = near_thresh
    data_geo['far_thresh'] = far_thresh
    data_geo['max_pixel_value'] = np.log(max_pixel_value)

    # scale the values that need scaling
    data_geo["DSD"] *= scale_factor
    data_geo["DSO"] *= scale_factor
    data_geo["dDetector"] = (np.array(data_geo["dDetector"]) * scale_factor).tolist()
    data_geo['nVoxel'] = data_geo['nVoxel'].tolist()
    data_geo["dVoxel"] = (np.array(data_geo["dVoxel"]).astype('float') * scale_factor).tolist()
    data_geo["offOrigin"] = (np.array(data_geo["offOrigin"]) * scale_factor).tolist()
    data_geo["offDetector"] = (np.array(data_geo["offDetector"]) * scale_factor).tolist()

    with open(f'{train_folder_name}general.json', 'w') as fp:
        json.dump(data_geo, fp)