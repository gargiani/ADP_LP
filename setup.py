from setuptools import setup, find_packages

install_reqs = ['torch==1.7.0']

setup(
        # Needed to silence warnings (and to be a worthwhile package)
        name='ADP_LP',
        url='https://github.com/gargiani/ADP_LP',
        author='Matilde Gargiani, Andrea Martinelli',
        author_email='gmatilde@ethz.ch, andremar@ethz.ch',
        packages=find_packages(),
        #requierements
        install_requires=install_reqs,
        # *strongly* suggested for sharing
        version='0.1',
        # The license can be anything you like
        license='MIT',
        #keywords
        keywords=['Linear Programming Approach', 'ADP', 'Data-Driven Optimal Control', 'Q-learning'],
        #description of package
        description='An implementation of the LP approach for Q-learning.'
)
