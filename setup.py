from setuptools import setup, find_packages

setup(name='EISD',
      version='0.2',
      description='Experimental Inferential Structure Determination',
      url='https://github.com/davas301/EISD.git',
      author='David Brookes',
      author_email='david.brookes@berkeley.edu',
      license='BSD',
      packages=find_packages(),
      scripts=['bin/eisdrun.py'],
      zip_safe=False)
