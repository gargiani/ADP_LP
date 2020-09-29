from setuptools import setup
setup(
        # Needed to silence warnings (and to be a worthwhile package)
        name='ADP_LP',
        url='https://github.com/gargiani/ADP_LP',
        author='Matilde Gargiani',
        author_email='gmatilde@ethz.ch, andremar@ethz.ch',
        packages=['gurobipy'],
        # Needed for dependencies
        install_requires=['numpy', 'scipy', 'torch', 'gurobipy', 'json', 'os'],
        # *strongly* suggested for sharing
        version='0.1',
        # The license can be anything you like
        license='MIT',
        description='An implementation of the LP approach for Q-learning.',
)

