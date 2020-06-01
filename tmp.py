import tensorflow as tf

from  lib.dataset.dataietr import DsfdDataIter


from train_config import config as cfg
# tf.enable_eager_execution()
train_ds=DsfdDataIter(cfg.DATA.root_path,cfg.DATA.train_txt_path,training_flag=True)
val_ds = DsfdDataIter(cfg.DATA.root_path,cfg.DATA.val_txt_path,training_flag=False)


trainds_iter = tf.data.Dataset.from_generator(train_ds,
                                            output_types=(tf.float32,tf.float32,tf.float32,tf.float32),
                                            output_shapes = ((None,None,None),
                                                             (None,None,None),
                                                             (None,None,None),
                                                             (None,None,None)) )

val_iter = tf.data.Dataset.from_generator(val_ds,
                                            output_types=(tf.float32,tf.float32,tf.float32,tf.float32),
                                            output_shapes = ((None,None,None),
                                                             (None,None,None),
                                                             (None,None,None),
                                                             (None,None,None)) )

def dataset_generator_fun_train(*args):
    return trainds_iter
def dataset_generator_fun_val(*args):
    return val_iter

# trainds_iter=trainds_iter.prefetch(tf.data.experimental.AUTOTUNE).batch(10)
trainds_iter=tf.data.Dataset.range(2).interleave(dataset_generator_fun_train,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                cycle_length=2
    ).batch(10).prefetch(tf.data.experimental.AUTOTUNE)

val_iter=tf.data.Dataset.range(2).interleave(dataset_generator_fun_val,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                cycle_length=2
    ).batch(10).prefetch(tf.data.experimental.AUTOTUNE)

for count_batch in trainds_iter:


    print(count_batch[0])
    print(count_batch[1].shape)
    print(count_batch[2].shape)
    print(count_batch[3].shape)
