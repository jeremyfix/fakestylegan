import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fakestylegan",
    version="1.0",
    author="Jeremy Fix",
    author_email="Jeremy.Fixcentralesupelec.fr",
    license="CCBY-NC-SA 4.0",
    description='A program to play with stylegan',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires = [
        'numpy==1.17.4',
        'dlib==19.22.1',
        'Pillow==7.0.0'
    ],
    python_requires='>=3.6',
)
