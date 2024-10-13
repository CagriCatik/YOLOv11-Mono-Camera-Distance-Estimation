from setuptools import setup, find_packages

setup(
    name='ml-depth-pro',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'torch',
        'ultralytics',
    ],
    description='Depth estimation with metric depth maps and object detection',
    author='Your Name',
    author_email='youremail@example.com',
)
