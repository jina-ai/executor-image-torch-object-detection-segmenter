# ✨ TorchObjectDetectionSegmenter

**TorchObjectDetectionSegmenter** is a class that ...

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites

Some conditions to fulfill before running the executor

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://TorchObjectDetectionSegmenter')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://TorchObjectDetectionSegmenter'
```

#### using source codes
Use the source codes from JinaHub in your python codes,

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://TorchObjectDetectionSegmenter')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://TorchObjectDetectionSegmenter'
```


### 📦️ Via Pypi

1. Install the `jinahub-executor-image-torch-object-detection-segmenter` package.

	```bash
	pip install git+https://github.com/jina-ai/executor-image-torch-object-detection-segmenter.git
	```

1. Use `jinahub-executor-image-torch-object-detection-segmenter` in your code

	```python
	from jina import Flow
	from jinahub.segmenter.torch_object_detection_segmenter import TorchObjectDetectionSegmenter
	
	f = Flow().add(uses=TorchObjectDetectionSegmenter)
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/EXECUTOR_REPO_NAME.git
	cd EXECUTOR_REPO_NAME
	docker build -t executor-image-torch-object-detection-segmenter .
	```

1. Use `executor-image-torch-object-detection-segmenter` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://executor-image-torch-object-detection-segmenter:latest')
	```
	

## 🎉️ Example 


```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://TorchObjectDetectionSegmenter')

with f:
    resp = f.post(on='foo', inputs=Document(), return_resutls=True)
	print(f'{resp}')
```

### Inputs 

`Document` with `blob` of the shape `256`.

### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape `embedding_dim` (=128, by default) with `dtype=nfloat32`.


## 🔍️ Reference
- Some reference

