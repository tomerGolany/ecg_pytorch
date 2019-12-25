"""
Install command: pip --user install -e .

"""

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['matplotlib', 'numpy', 'bokeh', 'tensorboardX', 'sklearn', 'torchvision',
                     'torch', 'wfdb', 'google-api-python-client', 'google-auth-httplib2',
                     'google-auth-oauthlib', 'opencv-python']

setup(name='ecg_pytorch',
      version='0.1',
      install_requires=REQUIRED_PACKAGES,
      description='Deep learning methods for ECG classifications',
      url='http://github.com/tomergolany/ecg_pytorch',
      author='Tomer Golany',
      author_email='tomer.golany@gmail.com',
      license='Technion',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
