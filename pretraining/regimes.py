from tasks.unmasking import UnmaskingRegressive
from tasks.classification import ClassificationRegressive
from tasks.complexity import ComplexityRegressive
from tasks.next import NextRegressive

class TrainingRegime():
    def __init__(self, name, param, dataset, testset, featuremap):
        self.name = name
        self.param = param
        self.dataset = dataset
        self.testset = testset
        self.featuremap = featuremap
        self.param["dim_features"] = len(featuremap)
        # pretraining tasks
        self.pretraining_tasks = {
            "unmasking": UnmaskingRegressive(self.param, self.featuremap, self.dataset, self.testset),
            "classification": ClassificationRegressive(self.param, self.featuremap, self.dataset, self.testset),
            "complexity": ComplexityRegressive(self.param, self.featuremap, self.dataset, self.testset),
            "next": NextRegressive(self.param, self.featuremap, self.dataset, self.testset),
        }
        if param["leave_one_out"] in self.pretraining_tasks:
            self.pretraining_tasks.pop(param["leave_one_out"])
        from models.model import PretrainModel
        output_heads = {}
        output_required = {}
        for key, task in self.pretraining_tasks.items():
            output_heads[key] = task.get_model_head()
            output_required[key] = True
        self.model = PretrainModel(self.param, self.name, output_heads, output_required)
        if self.param["model"] != "new":
            print("continue training...")
            self.model = self.model.load(self.param["model"])

    def predict(self, x, output_key):
        prediction = self.model(x)[output_key]
        return prediction
    def forward_train(self, x, y, batch_i, epoch_i):
        raise NotImplementedError(f"train forward not implemented.")
    def forward_valid(self, x, y, batch_i, epoch_i):
        raise NotImplementedError(f"valid forward not implemented.")    
    def evaluate_train_batch(self):
        raise NotImplementedError(f"train batch evaluation not implemented.")
    def evaluate_train_epoch(self):
        raise NotImplementedError(f"train epoch evaluation not implemented.")
    def evaluate_valid_batch(self):
        raise NotImplementedError(f"valid batch evaluation not implemented.")
    def evaluate_valid_epoch(self):
        raise NotImplementedError(f"valid epoch evaluation not implemented.")

class InBatch(TrainingRegime):
    def __init__(self, param, dataset, testset, featuremap):
        super().__init__("inbatch", param, dataset, testset, featuremap)
        self.batch_task_losses = {}

    def forward_train(self, x, y, batch_i, epoch_i):
        loss = 0
        m = len(self.pretraining_tasks)
        n = len(y) // m
        assert n>=1
        for i, (key,task) in enumerate(self.pretraining_tasks.items()):
            begin = i*n
            end = begin+n
            x_task, y_task = task.prepare_batch(x[begin:end].clone(), y[begin:end].clone())
            if self.param["leave_one_out_feature"] == "mask":
                x_task[:,:,self.featuremap.mask] = 0
            y_pred = self.predict(x_task, key)
            y_true = y_task
            task_loss = task.loss_train(y_pred, y_true)
            loss += task_loss
            self.batch_task_losses[key] = task_loss.item()
        return loss

    def forward_valid(self, x, y, batch_i, epoch_i):
        loss = 0
        m = len(self.pretraining_tasks)
        n = len(y) // m
        assert n >= 1, f"batchsize must be of size minimum 4 not {len(y)}"
        for i, (key,task) in enumerate(self.pretraining_tasks.items()):
            begin = i*n
            end = begin+n
            x_task, y_task = task.prepare_batch(x[begin:end].clone(), y[begin:end].clone())
            y_pred = self.predict(x_task, key)
            y_true = y_task
            task_loss = task.loss_valid(y_pred, y_true)
            loss += task_loss
            self.batch_task_losses[key] = task_loss.item()
            task.validate(x_task, y_task, y_pred, y_true, self.model)
        return loss

    def evaluate_train_batch(self, batch_step):
        result = {"losses/batch/train_"+str(task): loss for task, loss in self.batch_task_losses.items()}
        for key, task in self.pretraining_tasks.items():
            task_batch_evaluation = task.evaluate_train_batch(batch_step)
            result |= task_batch_evaluation
        return result
    def evaluate_train_epoch(self, epoch, batch_count):
        result = {}
        for key, task in self.pretraining_tasks.items():
            task_epoch_evaluation = task.evaluate_train_epoch(epoch, batch_count)
            result |= task_epoch_evaluation
        return result
    def evaluate_valid_batch(self, batch_step):
        result = {"losses/batch/valid_"+str(task): loss for task, loss in self.batch_task_losses.items()}
        for key, task in self.pretraining_tasks.items():
            task_batch_evaluation = task.evaluate_valid_batch(batch_step)
            result |= task_batch_evaluation
        return result
    def evaluate_valid_epoch(self, epoch, batch_count):
        result = {}
        for key, task in self.pretraining_tasks.items():
            task_epoch_evaluation = task.evaluate_valid_epoch(epoch, batch_count)
            result |= task_epoch_evaluation
        return result

class InEpoch(TrainingRegime):
    def __init__(self, param, dataset, testset, featuremap):
        super().__init__("inepoch", param, dataset, testset, featuremap)
        self.batch_counter_train = -1
        self.batch_counter_valid = -1
        self.pretraining_tasks_keys = list(self.pretraining_tasks.keys())
        self.pretraining_tasks_count = len(self.pretraining_tasks_keys)

    def forward_train(self, x, y, batch_i, epoch_i):
        self.batch_counter_train += 1
        task_id = self.batch_counter_train % self.pretraining_tasks_count
        task_key = self.pretraining_tasks_keys[task_id]
        self.model.output_required = dict.fromkeys(self.model.output_required, False)
        self.model.output_required[task_key] = True
        x_task, y_task = self.pretraining_tasks[task_key].prepare_batch(x,y)
        if self.param["leave_one_out_feature"] == "mask":
            x_task[:,:,self.featuremap.mask] = 0
        y_pred = self.predict(x_task, task_key)
        y_true = y_task
        loss = self.pretraining_tasks[task_key].loss_train(y_pred, y_true)
        return loss

    def forward_valid(self, x, y, batch_i, epoch_i):
        self.batch_counter_valid += 1
        task_id = self.batch_counter_valid % self.pretraining_tasks_count
        task_key = self.pretraining_tasks_keys[task_id]
        self.model.output_required = dict.fromkeys(self.model.output_required, False)
        self.model.output_required[task_key] = True
        x_task, y_task = self.pretraining_tasks[task_key].prepare_batch(x,y)
        y_pred = self.predict(x_task, task_key)
        y_true = y_task
        loss = self.pretraining_tasks[task_key].loss_valid(y_pred, y_true)
        self.pretraining_tasks[task_key].validate(x_task, y_task, y_pred, y_true, self.model)
        return loss

    def evaluate_train_batch(self, batch_step):
        task_id = self.batch_counter_train % self.pretraining_tasks_count
        task_key = self.pretraining_tasks_keys[task_id]
        task_batch_evaluation = self.pretraining_tasks[task_key].evaluate_train_batch(batch_step)
        return task_batch_evaluation

    def evaluate_train_epoch(self, epoch, batch_count):
        result = {}
        n = self.pretraining_tasks_count
        end = self.batch_counter_train # 9
        begin = end - batch_count + 1# 5
        task_ocurrences_in_batch = [0]*n
        for i in range(begin, end+1):
            task_ocurrences_in_batch[i%n] += 1

        for pos, (key,task) in enumerate(self.pretraining_tasks.items()):
            task_ocurrences = task_ocurrences_in_batch[pos]
            assert task_ocurrences >= 1, f"not enough batches per epoch, minimum 1 batch of every task per epoch is required."
            task_epoch_evaluation = task.evaluate_train_epoch(epoch, task_ocurrences)
            result |= task_epoch_evaluation
        return result

    def evaluate_valid_batch(self, batch_step):
        task_id = self.batch_counter_valid % self.pretraining_tasks_count
        task_key = self.pretraining_tasks_keys[task_id]
        task_batch_evaluation = self.pretraining_tasks[task_key].evaluate_valid_batch(batch_step)
        return task_batch_evaluation

    def evaluate_valid_epoch(self, epoch, batch_count):
        result = {}
        n = self.pretraining_tasks_count
        end = self.batch_counter_valid # 9
        begin = end - batch_count + 1# 5
        task_ocurrences_in_batch = [0]*n
        for i in range(begin, end+1):
            task_ocurrences_in_batch[i%n] += 1
        for pos, (key,task) in enumerate(self.pretraining_tasks.items()):
            task_ocurrences = task_ocurrences_in_batch[pos]
            task_epoch_evaluation = task.evaluate_valid_epoch(epoch, task_ocurrences)
            result |= task_epoch_evaluation
        return result
    
class AtEpoch(TrainingRegime):
    def __init__(self, param, dataset, testset, featuremap):
        super().__init__("atepoch", param, dataset, testset, featuremap)
        self.epoch_counter = 0
        self.active_epoch = 0
        self.pretraining_tasks_keys = list(self.pretraining_tasks.keys())
        self.pretraining_tasks_count = len(self.pretraining_tasks_keys)

    def forward_train(self, x, y, batch_i, epoch_i):
        if self.active_epoch != epoch_i:
            self.epoch_counter += 1
            self.active_epoch = epoch_i
        task_id = self.epoch_counter % self.pretraining_tasks_count
        task_key = self.pretraining_tasks_keys[task_id]
        self.model.output_required = dict.fromkeys(self.model.output_required, False)
        self.model.output_required[task_key] = True
        x_task, y_task = self.pretraining_tasks[task_key].prepare_batch(x,y)
        y_pred = self.predict(x_task, task_key)
        y_true = y_task
        loss = self.pretraining_tasks[task_key].loss_train(y_pred, y_true)
        return loss

    def forward_valid(self, x, y, batch_i, epoch_i):
        if self.active_epoch != epoch_i:
            self.epoch_counter += 1
            self.active_epoch = epoch_i
        task_id = self.epoch_counter % self.pretraining_tasks_count
        task_key = self.pretraining_tasks_keys[task_id]
        self.model.output_required = dict.fromkeys(self.model.output_required, False)
        self.model.output_required[task_key] = True
        x_task, y_task = self.pretraining_tasks[task_key].prepare_batch(x,y)
        if self.param["leave_one_out_feature"] == "mask":
            x_task[:,:,self.featuremap.mask] = 0
        y_pred = self.predict(x_task, task_key)
        y_true = y_task
        loss = self.pretraining_tasks[task_key].loss_valid(y_pred, y_true)
        self.pretraining_tasks[task_key].validate(x_task, y_task, y_pred, y_true, self.model)
        return loss

    def evaluate_train_batch(self, batch_step):
        task_id = self.epoch_counter % self.pretraining_tasks_count
        task_key = self.pretraining_tasks_keys[task_id]
        task_batch_evaluation = self.pretraining_tasks[task_key].evaluate_train_batch(batch_step)
        return task_batch_evaluation

    def evaluate_train_epoch(self, epoch, batch_count):
        task_id = self.epoch_counter % self.pretraining_tasks_count
        task_key = self.pretraining_tasks_keys[task_id]
        task_epoch_evaluation = self.pretraining_tasks[task_key].evaluate_train_epoch(epoch, batch_count)
        return task_epoch_evaluation

    def evaluate_valid_batch(self, batch_step):
        task_id = self.epoch_counter % self.pretraining_tasks_count
        task_key = self.pretraining_tasks_keys[task_id]
        task_batch_evaluation = self.pretraining_tasks[task_key].evaluate_valid_batch(batch_step)
        return task_batch_evaluation

    def evaluate_valid_epoch(self, epoch, batch_count):
        task_id = self.epoch_counter % self.pretraining_tasks_count
        task_key = self.pretraining_tasks_keys[task_id]
        task_epoch_evaluation = self.pretraining_tasks[task_key].evaluate_valid_epoch(epoch, batch_count)
        return task_epoch_evaluation

class AtTask(TrainingRegime):
    def __init__(self, param, dataset, testset, featuremap):
        super().__init__("attask", param, dataset, testset, featuremap)
        if param["task"] == None:
            raise ValueError("no task defined for training-regime attask.")
        else:
            self.task_name = param["task"]
        if self.task_name in self.pretraining_tasks:
            self.task = self.pretraining_tasks[self.task_name]
        else:
            raise ValueError(f"task {self.task_name} is not a valid pretraining task.")
        self.model.output_required = dict.fromkeys(self.model.output_required, False)
        self.model.output_required[self.task_name] = True

    def forward_train(self, x, y, batch_i, epoch_i):
        x_task, y_task = self.task.prepare_batch(x, y)
        if self.param["leave_one_out_feature"] == "mask":
            x_task[:,:,self.featuremap.mask] = 0
        y_pred = self.predict(x_task, self.task_name)
        y_true = y_task
        loss = self.task.loss_train(y_pred, y_true)
        return loss

    def forward_valid(self, x, y, batch_i, epoch_i):
        x_task, y_task = self.task.prepare_batch(x, y)
        y_pred = self.predict(x_task, self.task_name)
        y_true = y_task
        loss = self.task.loss_valid(y_pred, y_true)
        self.task.validate(x_task, y_task, y_pred, y_true, self.model)
        return loss

    def evaluate_train_batch(self, batch_step):
        task_batch_evaluation = self.task.evaluate_train_batch(batch_step)
        return task_batch_evaluation

    def evaluate_train_epoch(self, epoch, batch_count):
        task_epoch_evaluation = self.task.evaluate_train_epoch(epoch, batch_count)
        return task_epoch_evaluation

    def evaluate_valid_batch(self, batch_step):
        task_batch_evaluation = self.task.evaluate_valid_batch(batch_step)
        return task_batch_evaluation

    def evaluate_valid_epoch(self, epoch, batch_count):
        task_epoch_evaluation = self.task.evaluate_valid_epoch(epoch, batch_count)
        return task_epoch_evaluation

def get_regime(param:dict, dataset, testset, featuremap):
    regime = param["regime"]
    if regime == "inbatch":
        return InBatch(param, dataset, testset, featuremap)
    elif regime == "inepoch":
        return InEpoch(param, dataset, testset, featuremap)
    elif regime == "atepoch":
        return AtEpoch(param, dataset, testset, featuremap)
    elif regime == "attask":
        return AtTask(param, dataset, testset, featuremap)
    else:
        raise ValueError(f"regime {regime} not supported.")