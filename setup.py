from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
  name="miso",
  version="0.1.0",
  description="Resolving tissue complexity by multi-modal spatial omics modeling with MISO",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/kpcoleman/miso",
  author="Kyle Coleman",
  author_email="kpcoleman87@gmail.com",
  packages=find_packages(),
  #include_package_data=True,
  python_requires="==3.7.*",
  package_data = {'miso': ['checkpoints/*.pth'],},
  install_requires=["scikit-learn==1.0.2","scikit_image==0.19.3","torch==1.13.1","torchvision==0.14.1","numpy==1.21.6","Pillow>=6.1.0","opencv-python==4.6.0.66","scipy==1.7.3","einops==0.6.0","scanpy==1.9.1","tqdm==4.64.1"],
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Operating System :: OS Independent",
  ]
)


