import os
from setuptools import setup, find_packages, Command

import reinforce

def get_reqs(test=False):
    # optparse is included with Python <= 2.7, but has been deprecated in favor
    # of argparse.  We try to import argparse and if we can't, then we'll add
    # it to the requirements
    reqs = [
        'PyYaml',
    ]
    if test:
        # These take a long time to compile, and are only used for testing,
        # so we won't require them for a normal installation.
        reqs.extend([
            'PySide',
            'matplotlib',
            'numpy',
            'scipy',
        ])
    return reqs

class TestCommand(Command):
    description = "Runs unittests."
    user_options = [
        ('name=', None,
         'Name of the specific test to run.'),
        ('virtual-env-dir=', None,
         'The location of the virtual environment to use.'),
        ('pv=', None,
         'The version of Python to use. e.g. 2.7 or 3'),
    ]
    
    def initialize_options(self):
        self.name = None
        self.virtual_env_dir = './.env%s'
        self.pv = 0
        self.versions = [
            2.7,
            #3,
        ]
        
    def finalize_options(self):
        pass
    
    def build_virtualenv(self, pv):
        virtual_env_dir = self.virtual_env_dir % pv
        kwargs = dict(virtual_env_dir=virtual_env_dir, pv=pv)
        if not os.path.isdir(virtual_env_dir):
            cmd = ('virtualenv -p /usr/bin/python{pv} '
                '{virtual_env_dir}').format(**kwargs)
            #print(cmd)
            os.system(cmd)
            
            cmd = ('. {virtual_env_dir}/bin/activate; easy_install '
                '-U distribute; deactivate').format(**kwargs)
            os.system(cmd)
            
            for package in get_reqs(test=True):
                kwargs['package'] = package
                cmd = ('. {virtual_env_dir}/bin/activate; pip install '
                    '-U {package}; deactivate').format(**kwargs)
                #print(cmd)
                os.system(cmd)
    
    def run(self):
        versions = self.versions
        if self.pv:
            versions = [self.pv]
        
        for pv in versions:
            
            self.build_virtualenv(pv)
            kwargs = dict(pv=pv, name=self.name)
                
            if self.name:
                cmd = ('. ./.env{pv}/bin/activate; '
                    'python reinforce/test.py Tests.{name}; deactivate'
                ).format(**kwargs)
            else:
                cmd = ('. ./.env{pv}/bin/activate; '
                    'python reinforce/test.py Tests; deactivate'
                ).format(**kwargs)
                
            print(cmd)
            ret = os.system(cmd)
            if ret:
                return

setup(
    name='reinforce',
    version=reinforce.__version__,
    description='A simple Python reinforcement learning library.',
    author='Chris Spencer',
    author_email='chrisspen@gmail.com',
    url='https://github.com/chrisspen/reinforce',
    license='LGPL License',
    packages=find_packages(),
    #https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        #'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: LGPL License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ],
    zip_safe=False,
    install_requires=get_reqs(),
    cmdclass={
        'test': TestCommand,
    },
)
