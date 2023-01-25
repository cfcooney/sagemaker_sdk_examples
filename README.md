# sagemaker_sdk_examples
Repository of examples for running training and deployment of deep learning models on AWS with Amazon SageMaker SDK.

Working with HuggingFace and PyTorch models alongside SageMaker tools such as [Estimators](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html), [Hyperparameter Tuners](https://sagemaker.readthedocs.io/en/stable/api/training/tuner.html), [Experiments](https://docs.aws.amazon.com/sagemaker/latest/dg/experiments.html), and [Pipelines](https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html).

## Fine-tune HuggingFace models with SageMaker HuggingFace Estimator

* huggingface_estimator.ipynb uses HuggingFace Estimator with the HuggingFace [Trainer class](https://huggingface.co/docs/transformers/main_classes/trainer).
* estimator_for_bespoke.ipynb uses HuggingFace Estimator with a native PyTorch training script, allowing more control over aspects of the training procedure.

```python
for n in epochs:
  print(n)
```
