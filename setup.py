from setuptools import setup, find_packages

def get_requirements_txt():
    with open('requirements.txt', 'r') as file:
        return file.read().splitlines()
    
setup(
    name="pyprocess",
    version="0.1.0",
    packages=find_packages(exclude=['tests*']),
    install_requires= get_requirements_txt(),
    entry_points={
       'console_scripts': [
            'pyprocess=pyprocess.main:main',
        ],
    },
    author="Abdulkerim Akan",
    author_email="kerimakan77@gmail.com",
    description="A collection of multipurpose Python functions using image processing, OpenCV, and ML libraries.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kerim47/pyprocess",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    keywords="machine-learning, deep-learning, image-processing, pytorch, tensorflow, numpy, bounding-box, iou, computer-vision, cv",
)
