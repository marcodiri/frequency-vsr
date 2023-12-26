from lightning.pytorch.cli import LightningCLI

from data.datamodule import FolderDataModule
from models.gan_module import SRGAN
from models.networks.discr_nets import SpatialDiscriminator  # noqa: F401
from models.networks.fe_nets import FENet  # noqa: F401
from utils.lit_callbacks import ImageLog, MemProfiler  # noqa: F401


def cli_main():
    cli = LightningCLI(
        SRGAN,
        FolderDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()
