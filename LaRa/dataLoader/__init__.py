from dataLoader.gobjverse import gobjverse
from dataLoader.google_scanned_objects import GoogleObjsDataset
from dataLoader.instant3d import Instant3DObjsDataset
from dataLoader.mipnerf import MipNeRF360Dataset
from dataLoader.mvgen import MVGenDataset
from dataLoader.custom_loader import custom_loader
from dataLoader.housecat6d import housecat6d

dataset_dict = {'gobjeverse': gobjverse, 
                'GSO': GoogleObjsDataset,
                'instant3d': Instant3DObjsDataset,
                'mipnerf360': MipNeRF360Dataset,
                'mvgen': MVGenDataset,
                'custom': custom_loader,
                'housecat6d': housecat6d
                }