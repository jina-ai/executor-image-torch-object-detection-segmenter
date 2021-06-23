__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import setuptools

setuptools.setup(
    name='jinahub-torch-object-detection-segmenter',
    version='1',
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description='Torch Object Detection Segmenter',
    url='https://github.com/executor-image-torch-object-detection-segmenter',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    py_modules=['jinahub.segmenter.torch_object_detection_segmenter'],
    package_dir={'jinahub.segmenter': '.'},
    install_requires=open('requirements.txt').readlines(),
    python_requires='>=3.7',
)
