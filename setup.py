from setuptools import setup, find_packages
print("PACKAGES:",find_packages('src')+find_packages('./'))
setup(
  name='rpl_pybullet_sample_env',
  version='0.2.0',
  author='Dennis Hadjivelichkov and Sicelukwanda Zwane',
  author_email='ucabds6@ucl.ac.uk',
  packages= ['rpl_pybullet_sample_env','robot_models'],
  package_dir={'':'src'},
  url='http://pypi.python.org/pypi/rpl_pybullet_sample_env/',
  license='LICENSE.txt',
  description='A sample PyBullet environment exposing functions for robot control.',
  long_description=open('README.md').read(),
  install_requires=[
      "pybullet==3.2.5",
      "gym==0.23.1",
      "pykin==1.2.0",
      "numpy",
      "matplotlib"
  ],
)
