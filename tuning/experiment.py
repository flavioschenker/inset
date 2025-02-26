import torch
import wandb
from timeit import default_timer as timer
from benchmark.optimizer import Optimizer
from data.settings import device, deftype
from tasks.task import get_task
from models.model import TuningModel
from torchsummary import summary

class Experiment():
    def __init__(self, param:dict, dataset, testset):
        self.param = param
        self.jobid = param["jobid"]
        if param["debug"]:
            self.epochs = param["debug_epochs"]
            self.train_batch_size = param["debug_train_batch_size"]
            self.valid_batch_size = param["debug_valid_batch_size"]
        else:
            self.epochs = param["epochs"]
            self.train_batch_size = param["train_batch_size"]
            self.valid_batch_size = param["valid_batch_size"]
        self.featuremap = dataset.featuremap
        self.param["dim_features"] = len(self.featuremap)
        self.task = get_task(self.param, self.featuremap, dataset, testset)
        dim_dataset = len(dataset)
        dim_testset = len(testset)
        self.train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.train_batch_size, drop_last=True, shuffle=True)
        self.valid_dataloader = torch.utils.data.DataLoader(testset, batch_size=self.valid_batch_size, drop_last=True, shuffle=True)
        self.train_batch_count = dim_dataset // self.train_batch_size
        self.valid_batch_count = dim_testset // self.valid_batch_size

        self.model_head = self.task.get_model_head()
        self.model = TuningModel(self.param, self.model_head)
    
    def run(self):
        print(f"New Run: Id {self.jobid}, Task {self.task.name}, Model {self.param['model']}, Device {device}, DType: {torch.get_default_dtype()}")
        self.model.to(device)
        if self.param["verbosity"] > 2:
            summary(model=self.model, input_size=(50,20), dtypes=[torch.long])
        if self.param["wandbosity"]:
            wandb.watch(self.model, log_freq=100)
            wandb.define_metric("epoch")
            wandb.define_metric("batch")

        optimizer, scheduler = Optimizer(self.param, self.model)()

        for epoch in range(self.epochs):
            print(f"{'='*43} Epoch {epoch+1:>2}/{self.epochs:<2} {'='*44}")
            if self.param["train_model"]:
                self.train(epoch, optimizer)
            if self.param["eval_model"]:
                print(f"{'-'*100}")
                self.test(epoch)
            if self.param["train_model"]:
                scheduler.step()
        print(f"{'='*47} Done {'='*47}")

    def train(self, epoch, optimizer):
        self.model.train()
        epoch_loss = 0
        epoch_elapsed_time = 0

        for index, batch in enumerate(self.train_dataloader):
            batchtime_begin = timer()
            batch_step = index + epoch*self.train_batch_count
            if self.param["verbosity"] > 1:
                print(f"batch {index+1:>5}/{self.train_batch_count}", end=", ")

            # forward
            x, y = batch
            del batch
            x_task, y_task = self.task.prepare_batch(x.to(device), y.to(device))
            del x, y
            y_pred, y_true = self.task.predict(x_task, y_task, self.model)
            del x_task, y_task
            batch_loss = self.task.loss_train(y_pred, y_true)
            del y_pred, y_true
            # backwards
            epoch_loss += batch_loss.item()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            # training metrics
            batch_train_metrics = self.task.evaluate_train_batch(batch_step)
            if self.param["wandbosity"] > 1:
                wandb.define_metric("losses/batch/train_total", step_metric="batch")
                wandb.log({"losses/batch/train_total": batch_loss.item(), "batch": batch_step+1})
                batch_train_metrics["batch"] = batch_step+1
                for metric in batch_train_metrics:
                    wandb.define_metric(metric, step_metric="batch")
                wandb.log(batch_train_metrics)

            batchtime_end = timer()
            batchtime_total = batchtime_end-batchtime_begin
            epoch_elapsed_time += batchtime_total
            average_batchtime = epoch_elapsed_time/(index+1)
            remaining_epoch_time = (self.train_batch_count-index+1)*average_batchtime
            if self.param["verbosity"] > 1:
                print(f"loss {batch_loss:20.3f}, batchtime {batchtime_total:7.2f}s, remaining {remaining_epoch_time:7.0f}s", end="\r")
            del batch_loss, batch_train_metrics

        epoch_loss /= self.train_batch_count
        epoch_train_metrics = self.task.evaluate_train_epoch(epoch, self.train_batch_count)

        if self.param["wandbosity"]:
            wandb.define_metric("losses/epoch/train_total", step_metric="epoch")
            wandb.log({"losses/epoch/train_total": epoch_loss, "epoch": epoch+1})
            for metric in epoch_train_metrics:
                wandb.define_metric(metric, step_metric="epoch")
            epoch_train_metrics["epoch"] = epoch+1
            wandb.log(epoch_train_metrics)

        if self.param["verbosity"]:
            print(f"total training epoch loss {epoch_loss:74.3f}")

        # save model
        if not self.param["debug"] and not self.param["sweep"] and self.param["save_model"]:
            self.model.save(self.jobid)
            print(f"model state saved.")

    def test(self, epoch):
        with torch.no_grad():
            self.model.eval()
            epoch_loss = 0
            epoch_elapsed_time = 0

            for index, batch in enumerate(self.valid_dataloader):
                batchtime_begin = timer()
                batch_step = index + epoch*self.valid_batch_count
                if self.param["verbosity"] > 1:
                    print(f"batch {index+1:>5}/{self.valid_batch_count}", end=", ")
                
                # forward
                x, y = batch
                del batch
                x_task, y_task = self.task.prepare_batch(x.to(device), y.to(device))
                del x, y
                y_pred, y_true = self.task.predict(x_task.clone(), y_task.clone(), self.model)
                batch_loss = self.task.loss_valid(y_pred, y_true)
                self.task.validate(x_task, y_task, y_pred, y_true, self.model)
                del y_pred, y_true
                epoch_loss += batch_loss.item()

                # validation metrics
                batch_valid_metrics = self.task.evaluate_valid_batch(batch_step)
                if self.param["wandbosity"] > 1:
                    wandb.define_metric("losses/batch/valid_total", step_metric="batch")
                    wandb.log({"losses/batch/valid_total": batch_loss.item(), "batch": batch_step+1})
                    batch_valid_metrics["batch"] = batch_step+1
                    for metric in batch_valid_metrics:
                        wandb.define_metric(metric, step_metric="batch")
                    wandb.log(batch_valid_metrics)

                batchtime_end = timer()
                batchtime_total = batchtime_end-batchtime_begin
                epoch_elapsed_time += batchtime_total
                average_batchtime = epoch_elapsed_time/(index+1)
                remaining_epoch_time = (self.valid_batch_count-index+1)*average_batchtime
                if self.param["verbosity"] > 1:
                    print(f"loss {batch_loss:20.3f}, batchtime {batchtime_total:7.2f}s, remaining {remaining_epoch_time:7.0f}s", end="\r")
                del batch_loss, batch_valid_metrics

            epoch_loss /= self.valid_batch_count
            epoch_valid_metrics = self.task.evaluate_valid_epoch(epoch, self.valid_batch_count)

            if self.param["wandbosity"]:
                wandb.define_metric("losses/epoch/valid_total", step_metric="epoch")
                wandb.log({"losses/epoch/valid_total": epoch_loss, "epoch": epoch+1})
                epoch_valid_metrics["epoch"] = epoch + 1
                for metric in epoch_valid_metrics:
                    wandb.define_metric(metric, step_metric="epoch")
                wandb.log(epoch_valid_metrics)

            if self.param["verbosity"]:
                print(f"total validation epoch loss {epoch_loss:72.3f}")