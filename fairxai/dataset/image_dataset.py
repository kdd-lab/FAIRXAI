from lore_sa.dataset import Dataset
from pandas import DataFrame


class ImageDataset(Dataset):
    # FIXME: Per le immagini non avrò un dataset credo, il descrittore come dovrebbe essere fatto?

    def __init__(self, data: DataFrame, class_name: str = None, categorial_columns:list = None, ordinal_columns:list = None):

        self.class_name = class_name
        self.df = data

        # target columns forced to be the last column of the dataset
        if class_name is not None:
            self.df = self.df[[x for x in self.df.columns if x != class_name] + [class_name]]

        self.descriptor = {'numeric': {}, 'categorical': {}, 'ordinal': {}}

        # creation of a default version of descriptor
        self.update_descriptor(categorial_columns=categorial_columns, ordinal_columns=ordinal_columns)

    def update_descriptor(self, categorial_columns:list = None, ordinal_columns:list = None):
        pass