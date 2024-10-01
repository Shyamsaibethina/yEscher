from setuptools import setup, find_packages
setup(
    name="yescher",              
    version="0.2.21",              
    packages=find_packages(where='src'),     
    include_package_data=True, 
    package_dir={"": "src"},
    author='Shyam Sai Bethina',
    author_email='shyamsaibethina@gmail.com',
    url='https://github.com/Shyamsaibethina/yEscher',
    install_requires=[
    'beautifulsoup4',
    'Bio',
    'bokeh<2.5.0,>=2.4.0',
    'cobra<=0.24',
    'gurobipy',
    'joblib',
    'numpy',
    'optlang',
    'pandas',
    'pymysql',
    'pytest',
    'pytfa',
    'python-dotenv',
    'PyYAML',
    'rdkit',
    'Requests',
    'scikit_learn',
    'scipy',
    'setuptools',
    'sphinx_rtd_theme',
    'SQLAlchemy',
    'sympy',
    'tqdm',
    'pyqt5<5.13',
    'pyqtwebengine<5.13',
    'pathlib',
    'nbconvert>=7.11.0,<8.0.0',
    'pluggy>=1.0.0',
    'xlrd<2.0'
    ],
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',
    description='Run gene knockouts and analyize S. Cerevisae, built on top of yeastGEM',
    keywords=['yeast', 'escher', 'pytfa','ME models','thermodynamics','flux analysis','expression'],

    license='MIT',

    # See https://PyPI.python.org/PyPI?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Environment :: Console',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.9',
    ],
)
