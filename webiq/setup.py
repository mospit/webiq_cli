from setuptools import setup, find_packages

setup(
    name='webiq',
    version='0.1.0',
    packages=find_packages(where='.'),
    package_dir={'': '.'},
    install_requires=[
        # Add dependencies here, e.g.,
        # 'stagehand',
        # 'google-generativeai',
        # 'typer',
        # 'PyYAML',
    ],
    entry_points={
        'console_scripts': [
            'webiq=webiq.cli.main:app', # Assuming Typer app
        ],
    },
    author='[Your Name]',
    author_email='[Your Email]',
    description='AI-Powered Web Automation CLI Tool',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='[Your Project URL]', # e.g., GitHub repo
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Or your chosen license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
