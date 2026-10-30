import logging

logger = logging.getLogger("base")


def create_model(opt):
    model = opt["model"]

    if model == "denoising":
        from .denoising_model import DenoisingModel as M
    elif model == "correction":
        from .correction_model import CorrectionModel as M
    else:
        raise NotImplementedError("Model [{:s}] not recognized.".format(model))
    m = M(opt)
    logger.info("Model [{:s}] is created.".format(m.__class__.__name__))
    return m
