# gpt-mini

<img src="dalle.png" alt="A speedboat stopped by a futuristic cyborg, cyberpunk style." width="250">

##### *This image has been generated using OpenAI Dall-e 2.

<br /> This repository containts a minimalistic [Tensorflow](https://www.tensorflow.org/) (re-)re-implementation highly inspired to [Karpathy's minGPT](https://github.com/karpathy/minGPT) Pytorch re-implementation of the [OpenAI GPT](https://github.com/openai/gpt-2).
This code is intended for research and educative purposes, and should be treaded accordingly.

* [gpt/](gpt) contains the actual model implementation ([gpt/modeling.py](gpt/modeling.py)) and the code for running trainings ([gpt/trainer.py](gpt/trainer.py)).

## Setup

```
# Clone the repo.
git clone https://github.com/aitechnologies-it/gpt-mini
cd gpt-mini

# Make a python environment.
# eg. conda, pyenv

# Prepare pip.
# conda install pip
pip install --upgrade pip

# Install requirements.
pip install -r requirements.txt
```

## Examples

Example python notebooks can be found in the main directory. We currently provide [play_text.ipynb](play_text.ipynb) to train (both token- and char-level) GPT to learn generate text from text provided as input. Check also [train_tokenizer.ipynb](train_tokenizer.ipynb) that shows how to train an Huggingface Tokenizer on your own data.
Also, we provide [play_image.ipynb](play_image.ipynb) to train the model to generate cifar-10 images in an auto-regressive (pixel-level) fashion. 

## Usage

```python
import tensorflow as tf

from gpt.modeling import (GPT1Config, GPT)
from gpt.trainer import (TrainerConfig, Trainer)

class MyDataset(tf.data.Dataset):
    def _gen_examples_from(
        data: tf.Tensor, ...
    ):
        def _gen():
            for example in data:
                ...
                yield ...
        return _gen

    def __new__(
        cls, inputs: tf.Tensor, block_size: int, batch_size: int, ...
    ):
        dataset =  (
            tf.data.Dataset.from_generator(
                cls._gen_examples_from(data=inputs, ...),
                output_signature=(
                    tf.TensorSpec(shape=(block_size,), dtype=tf.int32),
                    tf.TensorSpec(shape=(block_size,), dtype=tf.int32))
                )
                .batch(batch_size, drop_remainder=True)
                .repeat()
                .prefetch(tf.data.experimental.AUTOTUNE)
                ...
        )
        return dataset


config = GPT1Config(
    vocab_size=128, block_size=1024,
    n_layer=3, n_head=3, n_embd=48
)
tconf = TrainerConfig(
    max_epochs=3, batch_size=64, learning_rate=0.003,
    do_lr_decay=False, warmup_ratio=0.1, cosine_decay_alpha=0.0, weight_decay=0.0,
    total_number_optimization_steps=total_number_optimization_steps, log_every_steps=10,
    ckpt_path='./logs', trial_id='my_trial_id'
)

model = GPT(config)


trainer = Trainer(
    model, dataset, total_number_optimization_steps, config=tconf
)

trainer.train()

```
