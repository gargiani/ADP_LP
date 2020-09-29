from setuptools import setup
import setuptools
setup(
        # Needed to silence warnings (and to be a worthwhile package)
        name='ADP_LP',
        url='https://github.com/gargiani/ADP_LP',
        author='Matilde Gargiani',
        author_email='gmatilde@ethz.ch, andremar@ethz.ch',
        # Needed for dependencies
        packages=setuptools.find_packages(),
        # *strongly* suggested for sharing
        version='0.1',
        # The license can be anything you like
        license='MIT',
        description='An implementation of the LP approach for Q-learning.',
)

