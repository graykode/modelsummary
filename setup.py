from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup_info = dict(
    name='modelsummary',
    version='1.1.1',
    author='Tae Hwan Jung(@graykode)',
    author_email='nlkey2022@gmail.com',
    url='https://github.com/graykode/modelsummary',
    description='All Model summary in PyTorch similar to `model.summary()` in Keras',
    long_description=long_description,
    long_description_content_type='text/markdown',  # This is important!
    license='MIT',
    install_requires=[ 'tqdm', 'torch', 'numpy'],
    keywords='pytorch model summary model.summary()',
    packages=["modelsummary"],
)

setup(**setup_info)