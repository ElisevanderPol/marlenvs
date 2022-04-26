from setuptools import setup

setup(name='marlenvs',
      version='0.2',
      install_requires=['gym', 'numpy'],
      description=('Multi Agent Reinforcement Learning environments '
                   'accompanying the paper "Multi-Agent MDP Homomorphic '
                   'Networks".'),
      packages=['marlenvs'],
      url='',
      license='MIT License',
      zip_safe=False)
