from setuptools import setup, find_packages
import os

def read_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()
        
setup(
    name='tippy',
    version='0.1.0',
    packages=find_packages(),  # Automatically finds the inner 'tippy'
    author='Hyeonho Choi',
    description='Transient Image Processing with python',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/hhchoi1022/tippy',  # optional
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=read_requirements(),  # ? This uses your requirements.txt

)

