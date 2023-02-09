# sagemaker_sdk_examples
Repository of examples for running training and deployment of deep learning models on AWS with Amazon SageMaker SDK.

Working with HuggingFace and PyTorch models alongside SageMaker tools such as [Estimators](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html), [Hyperparameter Tuners](https://sagemaker.readthedocs.io/en/stable/api/training/tuner.html), [Experiments](https://docs.aws.amazon.com/sagemaker/latest/dg/experiments.html), and [Pipelines](https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html).

## Fine-tune HuggingFace models with SageMaker HuggingFace Estimator

Fine-tuning DistilBERT on IMDB dataset using two different approaches to training:

* huggingface_estimator.ipynb uses HuggingFace Estimator with the HuggingFace [Trainer class](https://huggingface.co/docs/transformers/main_classes/trainer).
* estimator_for_bespoke.ipynb uses HuggingFace Estimator with a native PyTorch training script, allowing more control over aspects of the training procedure.

The bespoke_training.py script uses Pytorch DataLoaders and tqdm for logging progress:

```python
def _get_train_dataloader(dataset_dir, train_batch_size, data_collator=None, is_distributed=False, **kwargs):
    logger.info("Getting train dataloader")
    
    train_dataset = load_from_disk(dataset_dir)
    train_dataset = train_dataset.remove_columns("text")
    train_sampler = (
        data_dist.DistributedSampler(train_dataset) if is_distributed else None
    )
    return DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=data_collator,
        **kwargs,
    )
```
```python

with tqdm(train_dataloader, unit="batch") as training_epoch:
    for batch in training_epoch:

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        correct = training_performance(outputs, batch)
        accuracy = correct / args.train_batch_size

        training_epoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
```

## Perform hyperparameter tuning on HuggingFace models with SageMaker HyperparameterTuner

Blog post associated with this tuning available [here](https://ciaranfcooney.medium.com/hyperparameter-tuning-of-huggingface-models-with-aws-sagemaker-sdk-f727ac06cf36)

Optimize hyperparameter values when fine-tuning HuggingFace models:

* hyperparameter_tuning.ipynb uses HuggingFace Estimator, HyperparameterTuner, and training_script.py to optimize hp values. A 'Bayesian' tuning strategy is employed and the training script is native PyTorch (as above). Uses the IMDB dataset.
* hyperparameter_tuning_tweet_eval.ipynb uses HuggingFace Estimator, HyperparameterTuner, and training_script.py to optimize hp values. A 'Bayesian' tuning strategy is employed and the training script is native PyTorch (as above). Uses the tweet_eval dataset.
* tuning_with_hf_trainer.ipynb employs a 'Random' tuning strategy and relies on the HuggingFace Trainer class for training.

Tuning job results are analysed and the best model deployed for inference, and used to make predictions.

## SageMaker Experiments to track and analyse different training jobs

Track experimental runs in a systematic fashion with logging and analysis tools provided by AWS.
