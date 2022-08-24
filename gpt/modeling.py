import math

import tensorflow as tf
import tensorflow.keras as K
import tensorflow_addons as tfa


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class SchedulerFnWrapper(K.optimizers.schedules.LearningRateSchedule):
    def __init__(self, step_fn):
        self.step_fn = step_fn

    def __call__(self, step):
        return self.step_fn(step=step)


def get_initializer(mean=0.0, stddev=0.02):
    # return K.initializers.RandomNormal(mean=mean, stddev=stddev)
    return K.initializers.TruncatedNormal(mean=0.0, stddev=stddev)


class CausalSelfAttention(K.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.query = K.layers.Dense(
            units=config.n_embd, use_bias=True,
            kernel_initializer=get_initializer(),
            bias_initializer=K.initializers.Zeros(),
        )
        self.key = K.layers.Dense(
            units=config.n_embd, use_bias=True,
            kernel_initializer=get_initializer(),
            bias_initializer=K.initializers.Zeros(),
        )
        self.value = K.layers.Dense(
            units=config.n_embd, use_bias=True,
            kernel_initializer=get_initializer(),
            bias_initializer=K.initializers.Zeros(),
        )
        # regularization
        self.attn_drop = K.layers.Dropout(config.attn_pdrop)
        self.resid_drop = K.layers.Dropout(config.resid_pdrop)
        # output projection
        self.proj = K.layers.Dense(
            units=config.n_embd, use_bias=True,
            kernel_initializer=get_initializer(),
            bias_initializer=K.initializers.Zeros(),
        )

        self.config = config

    def split_heads(self, x, batch_size):
        new_shape = (batch_size, -1, self.config.n_head, self.config.n_embd // self.config.n_head)
        x = tf.reshape(x, shape=new_shape)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def attention_mask(self, nd, ns, *, dtype):
        """1's in the lower triangle, counting from the lower right corner.
        Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        i = tf.range(nd)[:,None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)

    def mask_attn_weights(self, w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = w.shape
        b = self.attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def call(self, inputs, training):
        B, T, C = inputs.shape

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(inputs) # (B, T, C)
        k = self.key(inputs)   # (B, T, C)
        v = self.value(inputs) # (B, T, C)

        q = self.split_heads(q, batch_size=B) # (B, nh, T, hs)
        k = self.split_heads(k, batch_size=B) # (B, nh, T, hs)
        v = self.split_heads(v, batch_size=B) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = tf.matmul(q, k, transpose_b=True) # (B, nh, T, T)
        att = att * tf.math.rsqrt(tf.cast(k.shape[-1], att.dtype))
        att = self.mask_attn_weights(att)
        att = tf.nn.softmax(att, axis=-1)
        att = self.attn_drop(att, training=training)

        y = tf.matmul(att, v) # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = tf.transpose(y, perm=[0, 2, 1, 3]) # (B, nh, T, hs) -> (B, T, nh, hs)
        y = tf.reshape(y, shape=(-1, T, C)) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y), training=training)
        return y
    

class Block(K.layers.Layer):
    """ an unassuming Transformer block """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = K.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = K.layers.LayerNormalization(epsilon=1e-6)
        self.attn = CausalSelfAttention(config)
        self.mlp = K.Sequential([
            K.layers.Dense(
                units=4*config.n_embd, use_bias=True, activation="gelu",
                kernel_initializer=get_initializer(),
                bias_initializer=K.initializers.Zeros(),
            ),
            K.layers.Dense(
                units=config.n_embd, use_bias=True,
                kernel_initializer=get_initializer(),
                bias_initializer=K.initializers.Zeros(),
            ),
            K.layers.Dropout(config.resid_pdrop)
        ])

    def call(self, x, training):
        x = x + self.attn(self.ln1(x), training=training)
        x = x + self.mlp(self.ln2(x), training=training)
        return x


class GPT(K.Model):
    """  the full GPT language model for language modeling """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # input embedding stem
        self.tok_emb = K.layers.Embedding(
            input_dim=config.vocab_size, output_dim=config.n_embd,
            embeddings_initializer=get_initializer(),
            name="token_embedding",
        )
        self.drop = K.layers.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = K.Sequential([Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = K.layers.LayerNormalization(epsilon=1e-6)
        self.head = K.layers.Dense(
            units=config.vocab_size, use_bias=False,
            kernel_initializer=get_initializer(),
        )


    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            "positional_embedding",
            shape=(1, self.config.block_size, self.config.n_embd),
            initializer=get_initializer(), dtype=tf.float32,
            trainable=True,
        )

    def call(self, input_ids, targets=None, training=False):
        _, T = input_ids.shape

        if T > self.config.block_size:
            raise ValueError("Cannot forward, model block size is exhausted.")

        # input embedding
        token_embeddings = self.tok_emb(input_ids)
        position_embeddings = self.pos_emb[:, :T, :]
        e = self.drop(token_embeddings + position_embeddings, training=training)
        h = self.blocks(e, training=training)
        h = self.ln_f(h, training=training)
        logits = self.head(h)
        loss = None
        if targets is not None:
            loss_fn = K.losses.SparseCategoricalCrossentropy(from_logits=True)
            loss = loss_fn(targets, logits)
        outputs = (logits, loss)
        return outputs

    def configure_optimizers(self, train_config):
        # def decayed_lr(init_decay_value):
        #     def _decayed_lr(step):
        #         if self.tokens < config.warmup_tokens:
        #             lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
        #         else:
        #             progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
        #             lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        #         # lr = config.learning_rate * lr_mult
        #         lr = init_decay_value * lr_mult
        #         return lr
        #     return _decayed_lr

        # def gpt_classic_decayed_lr(
        #     peak_lr, end_lr, global_step, warmup_steps, total_steps,
        # ):
        #     warmup_pct = tf.clip_by_value(global_step, 0, warmup_steps) / warmup_steps
        #     anneal_pct = tf.clip_by_value(global_step - warmup_steps, 0, total_steps) / total_steps
        #     return warmup_pct * peak_lr - (peak_lr - end_lr) * (1.0 - tf.math.cos(tf.constant(math.pi) * anneal_pct)) / 2.0
            
        # def gpt_karpathy_decayed_lr(init_lr, global_step, warmup_steps, decay_steps):
        #     init_lr = tf.cast(init_lr, dtype=tf.float32)
        #     global_step  = tf.cast(global_step, dtype=tf.float32)
        #     warmup_steps = tf.cast(warmup_steps, dtype=tf.float32)
        #     decay_steps  = tf.cast(decay_steps, dtype=tf.float32)
        #     progress = (global_step - warmup_steps) / (tf.maximum(1, decay_steps - warmup_steps))
        #     lr_mult = tf.maximum(0.1, 0.5 * (1.0 + tf.math.cos(tf.constant(math.pi, dtype=tf.float32) * progress)))
        #     lr = init_lr * lr_mult
        #     return lr

        def cosine_decayed_lr(global_step, decay_steps, alpha):
            global_step = tf.cast(global_step, dtype=tf.float32)
            decay_steps = tf.cast(decay_steps, dtype=tf.float32)
            global_step = tf.minimum(global_step, decay_steps)
            completed_fraction = global_step / decay_steps
            cosine_decayed = 0.5 * (1.0 + tf.cos(
                tf.constant(math.pi, dtype=completed_fraction.dtype) * completed_fraction))
            alpha = tf.cast(alpha, dtype=cosine_decayed.dtype)
            decayed = (1 - alpha) * cosine_decayed + alpha
            lr = tf.cast(train_config.learning_rate, dtype=decayed.dtype)
            return lr * decayed

        def decayed_lr(step):
            global_steps_int = tf.cast(step, tf.int32)
            warmup_steps_int = tf.constant(
                math.ceil(train_config.warmup_ratio * train_config.total_number_optimization_steps),
                dtype=tf.int32
            )
            nb_optimization_steps = tf.cast(train_config.total_number_optimization_steps, dtype=tf.int32)
            decay_steps_int = nb_optimization_steps - warmup_steps_int

            global_steps_float = tf.cast(global_steps_int, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            warmup_percent_done = global_steps_float / warmup_steps_float
            warmup_learning_rate = train_config.learning_rate * warmup_percent_done

            is_warmup = tf.cast(global_steps_int <= warmup_steps_int, tf.float32)
            # nb_optimization_steps = tf.cast(train_config.nb_optimization_steps, dtype=tf.float32)
            alpha = tf.constant(train_config.cosine_decay_alpha, dtype=tf.float32)
            learning_rate = cosine_decayed_lr(global_steps_int, decay_steps_int, alpha=alpha)
            learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
            return learning_rate

        if train_config.do_lr_decay:
            learning_rate_float_or_fn = SchedulerFnWrapper(step_fn=decayed_lr)
            weight_decay_float_or_fn  = SchedulerFnWrapper(step_fn=decayed_lr)
        else:
            learning_rate_float_or_fn = train_config.learning_rate
            weight_decay_float_or_fn  = train_config.weight_decay

        
        optimizer = tfa.optimizers.AdamW(
            weight_decay=weight_decay_float_or_fn,
            learning_rate=learning_rate_float_or_fn,
            beta_1=train_config.beta_1,
            beta_2=train_config.beta_2,
            global_clipnorm=train_config.grad_norm_clip,
            exclude_from_weight_decay=[
                "positional_embedding",
                "token_embedding",
                "LayerNorm", "layer_norm", "bias"
            ]
        )
        
        return optimizer

    def sample(
        self,
        input_ids,
        steps,
        temperature=1.0,
        sample=False,
        top_k=None
    ):
        # get model's context size
        ctx_sz = self.config.block_size
        
        for _ in range(steps):
            B, S = input_ids.shape
            input_ids_cond = input_ids
            if S > ctx_sz: # crop context if needed
                input_ids_cond = input_ids[:,-ctx_sz:]
            logits, _ = self(input_ids_cond, training=False)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                # optionally crop probabilities to only the top k options
                v, _ = tf.math.top_k(logits, top_k, sorted=True)
                logits = tf.identity(logits).numpy()
                logits[logits < v.numpy()[:, [-1]]] = -float('Inf')
            probabilities = tf.nn.softmax(logits, axis=-1)
            if sample:
                chunk_id = tf.random.categorical(tf.math.log(probabilities), num_samples=1)
            else:
                _, chunk_id = tf.math.top_k(probabilities, k=1)
            input_ids = tf.concat([
                input_ids, tf.reshape(tf.cast(chunk_id, dtype=input_ids.dtype), shape=(B, 1))], axis=-1
            )
        return input_ids

