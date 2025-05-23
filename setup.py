import pathlib
import setuptools


def parse_reqs(filename):
  requirements = pathlib.Path(filename)
  requirements = requirements.read_text().split('\n')
  requirements = [x for x in requirements if x.strip()]
  return requirements


setuptools.setup(
    name='dreamer-grpo',
    version='0.0.1',
    author='chenxingqiang',
    author_email='joy6677@qq.com',
    description='Mastering Diverse Domains through World Models with GRPO',
    url='http://github.com/chenxingqiang/dreamer-grpo',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=parse_reqs('requirements.txt'),
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
