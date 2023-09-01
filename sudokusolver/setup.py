from setuptools import setup, find_packages

setup(
    name="sudokusolver",
    license="MIT",
    version="0.1",
    description="A Sudoku Solver",
    author="Michael Atkinson",
    author_email="atki2828@gmail.com",
    packages=find_packages(),
    install_requires=[
        "absl-py==1.2.0",
        "astunparse==1.6.3",
        "cachetools==5.2.0",
        "fastjsonschema",
        "flatbuffers==1.12",
        "gast==0.4.0",
        "google-auth==2.11.0",
        "google-auth-oauthlib==0.4.6",
        "google-pasta==0.2.0",
        "grpcio==1.47.0",
        "h5py==3.7.0",
        "keras==2.9.0",
        "Keras-Preprocessing==1.1.2",
        "numpy==1.23.2",
        "protobuf==3.19.4",
        "tensorboard==2.9.1",
        "tensorflow==2.9.1",
        "tensorflow-estimator==2.9.0",
        "opencv-contrib-python==4.6.0.66",
        "matplotlib",  # Same for Matplotlib, adjust the version as necessary.
    ],
    # other arguments here...
)
