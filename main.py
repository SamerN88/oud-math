import os

from models import BezierModel, GeneralizedGDModel, GrowthDecayModel, StaticModel
from procedure import RibProcedure


# Oud dimensions in mm
H = 500
Z = 60


STATIC_MODEL = StaticModel(H=H, Z=Z)

GROWTH_DECAY_MODEL = GrowthDecayModel(
    H=H,
    Z=Z,
    alpha=0.449
)

GGD_MODEL = GeneralizedGDModel(
    H=H,
    Z=Z,
    alpha=0.421,
    beta=0.935,
    k=0.887
)

BEZIER_MODEL = BezierModel(
    H=H,
    Z=Z,
    x1=0, y1=0.396,
    x2=0.2, y2=0.496,
    x3=0.78, y3=0.32
)

MODELS = [
    STATIC_MODEL,
    GROWTH_DECAY_MODEL,
    GGD_MODEL,
    BEZIER_MODEL
]

RIB_PDF_DIR = 'rib-pdf-examples'
RIB_PDF_PATHS = {model.__class__.__name__: os.path.join(RIB_PDF_DIR, f'{model.__class__.__name__}_rib.pdf') for model in MODELS}


def test_all(num_ribs, plot=False):
    for model in MODELS:
        if plot:
            model.plot()
        procedure = RibProcedure(model, num_ribs, multiprocessing=True, verbose=True)
        procedure.run(RIB_PDF_PATHS[model.__class__.__name__])
        procedure.print()


def main():
    # TODO: consider how to allow users to perform different functions via Terminal
    # TODO: For publishing, maybe rename this function and file to "sandbox"? Not sure. Think about it.
    num_ribs = 17

    model = GeneralizedGDModel(H=500, Z=60, alpha=0.4631684, beta=0.9221139, k=1.03458806)
    procedure = RibProcedure(model, num_ribs, multiprocessing=True, verbose=True)
    procedure.run(os.path.join(RIB_PDF_DIR, 'test.pdf'))

    # test_all(num_ribs, plot=False)


if __name__ == '__main__':
    main()
