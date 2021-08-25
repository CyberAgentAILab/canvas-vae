import logging

logger = logging.getLogger(__name__)


def create_transform(dataset):
    if dataset == 'crello':
        from .generate_crello import GenerateDataset
        return GenerateDataset
    elif dataset == 'crello-document':
        from .generate_crello_document import GenerateDocumentDataset
        return GenerateDocumentDataset
    elif dataset == 'crello-image':
        from .generate_crello_image import GenerateImageDataset
        return GenerateImageDataset
    elif dataset == 'rico':
        from .generate_rico import GenerateRicoDataset
        return GenerateRicoDataset
    elif dataset == 'magazine':
        from .generate_magazine import GenerateMagazineDataset
        return GenerateMagazineDataset
    raise ValueError('invalid dataset: %s' % dataset)
