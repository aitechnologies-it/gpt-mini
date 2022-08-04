import os
import datetime

import tensorflow as tf
import tensorflow.keras as K

from tqdm import tqdm

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    beta_1 = 0.9
    beta_2 = 0.999
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    # final_tokens = 260e9 # (at what point we reach 10% of original LR)
    
    do_lr_decay = False
    warmup_ratio = 0.2
    cosine_decay_alpha = 0.0

    total_number_optimization_steps = None

    eval_every_steps = 1
    log_every_steps  = 1
    
    # checkpoint settings
    ckpt_path = './logs'
    trial_id = 'gtp-mini'

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(
        self, model, train_dataset, nb_optimization_steps, config, eval_dataset=None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.nb_optimization_steps = nb_optimization_steps
        self.config = config
        self.tokens = 0

    def train(self):
        model, config = self.model, self.config
        train_acc_metric = K.metrics.SparseCategoricalAccuracy()
        eval_acc_metric = K.metrics.SparseCategoricalAccuracy()
        eval_mean_loss_metric = K.metrics.Mean()
        eval_mean_acc_metric = K.metrics.Mean()
        
        optimizer = model.configure_optimizers(config)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(config.ckpt_path, f"{current_time}_{config.trial_id}", 'train')
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        
        if self.eval_dataset is not None:
            eval_log_dir = os.path.join(config.ckpt_path, f"{current_time}_{config.trial_id}", 'dev')
            eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

        self.tokens = 0

        @tf.function
        def train_step(data):
            input_ids, labels = data
            
            with tf.GradientTape() as tape:
                logits, loss = model(input_ids, labels, training=True)
            
            gradients = tape.gradient(loss, model.trainable_weights)

            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            train_acc_metric.update_state(labels, logits)
            return loss

        @tf.function
        def test_step(data):
            input_ids, labels = data
            logits, loss = model(input_ids, labels, training=False)
            eval_acc_metric.update_state(labels, logits)
            eval_mean_loss_metric.update_state([loss])
            eval_mean_acc_metric.update_state([eval_acc_metric.result()])
            return loss

        def update_nb_active_tokens_from(labels):
            labels_nz = int(tf.reduce_sum(tf.cast(labels >= 0, dtype=tf.int32)))
            self.tokens += labels_nz

        def get_learning_rate_at_step(step=None):
            lr = None
            if isinstance(optimizer.learning_rate, tf.Variable):
                lr = float(optimizer.learning_rate.numpy())
            elif isinstance(optimizer.learning_rate, K.optimizers.schedules.LearningRateSchedule):
                lr = K.backend.eval(optimizer.learning_rate(step))
            return lr

        pbar = tqdm(enumerate(self.train_dataset), total=self.nb_optimization_steps)
        for global_step, data in pbar:
            input_ids, labels = data
            loss_value = train_step((input_ids, labels))
            update_nb_active_tokens_from(labels)
            lr = get_learning_rate_at_step(global_step)

            progress_str = f"step {global_step}: loss {float(loss_value):.5f} - "\
                f"acc {float(train_acc_metric.result()*100):.2f}% - "\
                f"lr {float(lr):.6f}"
            pbar.set_description(progress_str)

            if self.eval_dataset is not None and global_step % config.eval_every_steps:
                pbar_dev = tqdm(enumerate(self.eval_dataset))
                for _, data in pbar_dev:
                    input_ids, labels = data
                    _ = test_step((input_ids, labels))

                with eval_summary_writer.as_default():
                    tf.summary.scalar('batch_loss', eval_mean_loss_metric.result(), step=global_step)
                    tf.summary.scalar('batch_accuracy', eval_mean_acc_metric.result()*100, step=global_step)
                    tf.summary.scalar('batch_perplexity', tf.exp(eval_mean_loss_metric.result()), step=global_step)

            if global_step % config.log_every_steps:
                with train_summary_writer.as_default():
                    tf.summary.scalar('batch_loss', float(loss_value), step=global_step)
                    tf.summary.scalar('batch_accuracy', train_acc_metric.result()*100, step=global_step)
                    tf.summary.scalar('batch_perplexity', tf.exp(loss_value), step=global_step)
                    tf.summary.scalar('learning_rate', lr, step=global_step)

            if global_step == self.nb_optimization_steps:
                break
        
        train_acc_metric.reset_states()
        eval_acc_metric.reset_states()
        eval_mean_loss_metric.reset_states()