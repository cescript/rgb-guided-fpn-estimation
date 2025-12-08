from models.SAFTAModel import SAFTAModel
from models.EMPTYModel import EMPTYModel
from models.BESTModel import BESTModel
from models.DLSNUCModel import DLSNUCModel
from models.D1WLSModel import D1WLSModel
from models.MULTIVIEWModel import MULTIVIEWModel
from models.DCGANModel import DCGANModel

# generate a FPN detector model with the given options
def GenerateModel(model_name):
    if model_name == 'SAFTA':
        return SAFTAModel(use_rgb=False, fpn_estimator="GRU")
    elif model_name == 'SAFTA-RGB':
        return SAFTAModel(use_rgb=True, fpn_estimator="GRU")
    elif model_name == 'SAFTA-RGB-OLS':
        return SAFTAModel(use_rgb=True, fpn_estimator="OLS")
    elif model_name == 'EMPTY':
        return EMPTYModel()
    elif model_name == 'BEST':
        return BESTModel()
    elif model_name == 'DLSNUC':
        return DLSNUCModel()
    elif model_name == 'D1WLS':
        return D1WLSModel()
    elif model_name == 'MULTIVIEW':
        return MULTIVIEWModel()
    elif model_name == 'DCGAN':
        return DCGANModel()
    else:
        raise ValueError(f"Model {model_name} not found")