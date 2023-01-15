import argparse
from cgi import test
from email.policy import default
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch.utils.data.distributed as data_dist
from datasets import load_from_disk
import evaluate
#from datasets.filesystems import S3FileSystem
import torch as th
from tqdm.auto import tqdm
import json

import logging
import sys
import os


#from utils import set_global_logging_level

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


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

def _get_test_dataloader(dataset_dir, test_batch_size, data_collator=None, **kwargs):
    logger.info("Getting test dataloader.")
    
    test_dataset = load_from_disk(dataset_dir)
    return DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        **kwargs,
    )


def train(args):
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    
    train_dataloader = _get_train_dataloader(args.training_dir, args.train_batch_size, data_collator=data_collator)
    test_dataloader = _get_test_dataloader(args.testing_dir, args.test_batch_size, data_collator=data_collator)
        
    logger.debug(f"Processes {len(train_dataloader.sampler)}/{len(train_dataloader.dataset)} ({100.0 * len(train_dataloader.sampler) / len(train_dataloader.dataset)}%) of train data")
    logger.debug(f"Processes {len(test_dataloader.sampler)}/{len(test_dataloader.dataset)} ({100.0 * len(test_dataloader.sampler) / len(test_dataloader.dataset)}%) of test data")
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    
    model.to(device)
        
    optimizer = AdamW(model.parameters(), lr=float(args.learning_rate))
    
    num_training_steps = args.epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )

    #progress_bar = tqdm(range(num_training_steps))

    model.train()
    for _ in range(args.epochs):
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
                #progress_bar.update(1)
            
            evaluate_model(model, test_dataloader, device)

    save_model(model, tokenizer, args.model_dir)
    logger.info(f"Made it this far")

def training_performance(outputs, batch):
    
    correct = 0
    labels = batch.pop("labels")
    logits = outputs.logits
    predictions = th.argmax(logits, dim=-1)
    correct += predictions.eq(labels.view_as(predictions)).sum().item()
    return correct

def evaluate_model(model, test_dataloader, device):
    logger.info(f"Evaluating...")

    metric = evaluate.load("glue", "mrpc")
    model.eval()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with th.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = th.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions,  references=batch.pop("labels"))
    logger.info(metric.compute())


def compute_loss(model, inputs, test_dataloader):
    model.eval()
    test_loss = 0
    loss_function = th.nn.NLLLoss()
    with th.no_grad():
        for batch in test_dataloader:
            outputs = model(**batch)
            logits = outputs.logits
            predictions = th.argmax(logits, dim=-1)
            test_loss += loss_function(predictions, batch.pop("labels"), size_average=False).item()
        
    test_loss /= len(test_dataloader.dataset)
    return test_loss




# def model_fn(model_dir):
    # device = th.device("cuda" if th.cuda.is_available() else "cpu")
    # #model = th.nn.DataParallel(NeuralNet())
    # with open(os.path.join(model_dir, "model.pth"), "rb") as f:
    #     model.load_state_dict(torch.load(f))
    # return model.to(device)


def save_model(model, tokenizer, model_dir):
    logger.info("Saving the model.")

    model.save_pretrained(model_dir, state_dict=model.cpu().state_dict())
    tokenizer.save_pretrained(model_dir)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs. Default=1")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training. Default=16")
    parser.add_argument("--test_batch_size", type=int, default=100, help="Batch size for testing. Default=100")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmmup steps. Default=500")
    parser.add_argument("--scheduler", type=str, default="linear", help="HuggingFace learning rate scheduler. Default='linear'")
    parser.add_argument("--learning_rate", type=str, default="5e-5", help="Learning rate. Default=5e-5")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Pretrained Huggingface model. Default='distilbert-base-uncased'")
    parser.add_argument("--tokenizer", type=str, default="distilbert-base-uncased", help="HuggingFace tokenizer. Default='distilbert-base-uncased'")
        
     # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])

    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--testing_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    
    args, _ = parser.parse_known_args()
    
    train(args)