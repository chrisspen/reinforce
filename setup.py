import os
from setuptools import setup, find_packages, Command

import reinforce

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

def get_reqs(*fns):
    lst = []
    for fn in fns:
        for package in open(os.path.join(CURRENT_DIR, fn)).readlines():
            package = package.strip()
            if not package:
                continue
            lst.append(package.strip())
    return lst

setup(
    name='reinforce-toolkit',
    version=reinforce.__version__,
    description='A simple Python reinforcement learning library.',
    author='Chris Spencer',
    author_email='chrisspen@gmail.com',
    url='https://github.com/chrisspen/reinforce',
    license='LGPL License',
    packages=find_packages(),
    #https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
    ],
    zip_safe=False,
    #TODO:revert once PyBrain officially releases Python3 fixes
#     install_requires=get_reqs('requirements.txt'),
#     tests_require=get_reqs('requirements-test.txt'),
)
