from distutils.core import setup
import reinforce
setup(name='reinforce',
    version=reinforce.__version__,
    description='A simple Python reinforcement learning library.',
    author='Chris Spencer',
    author_email='chrisspen@gmail.com',
    url='https://github.com/chrisspen/reinforce',
    license='LGPL License',
    packages=[
        'reinforce',
    ],
    #https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        #'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: LGPL License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ],
    install_requires = [
    ],
)