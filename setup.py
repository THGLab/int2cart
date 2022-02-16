import setuptools
from os import path


here = path.abspath(path.dirname(__file__))

# # Get the long description from the README file
# with open(path.join(here, 'README.md')) as f:
#     long_description = f.read()

long_description = None

if __name__ == "__main__":
    setuptools.setup(
        name='modelling',
        version='1.0.0',
        author='Jie Li',
        author_email='jerry-li1996@berkeley.edu',
        project_urls={
            'Source': 'https://github.com/THGLab/A-EISD',
            # 'url': 'https://THGLab.github.io/A-EISD/'
        },
        description=
        "Automated experimental inferential structure determination (A-EISD) of ensembles for intrinsically disordered proteins using deep learning generative models.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        keywords=[
            'Machine Learning', 'Data Mining', 'Proteins',
            'Structure Determination', "Generative Model",
            "Bayesian Framework"
        ],
        license='MIT',
        packages=setuptools.find_packages(),
        include_package_data=True,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Natural Language :: English',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
        ],
        install_requires=[
            'torch',
            'future'
        ],
        zip_safe=False,
    )