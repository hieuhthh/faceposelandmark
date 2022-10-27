import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def set_memory_growth():
    physical_devices = tf.config.list_physical_devices('GPU') 
    for gpu_instance in physical_devices: 
        tf.config.experimental.set_memory_growth(gpu_instance, True)

def auto_select_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print('Using TPU')
    except:
        try:
            strategy = tf.distribute.MirroredStrategy()
            using_gpus = True
            print('Using GPUs')
        except:
            strategy = tf.distribute.get_strategy()
            print('Using 1 GPU')

    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    
    return strategy
    
# basic + tensorflow
def seedEverything(seed):
    # tensorflow random seed 
    import tensorflow as tf 
    def seedTF(seed):
        tf.random.set_seed(seed)

    import random
    import os
    import numpy as np
    def seedBasic(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)

    seedBasic(seed)
    seedTF(seed)

def visual_save_metric(his, metric):
    val_metric = 'val_' + metric

    print(f'BEST {val_metric}:', np.min(his.history[val_metric]), 'at epoch:', np.argmin(his.history[val_metric]) + 1)

    plt.figure()
    plt.plot(his.history[metric], label=f'train {metric}')
    plt.plot(his.history[val_metric], label=f'test {metric}')
    plt.title(f'Plot History: Model {metric}')
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend([f'Train {metric}', f'Test {metric}'], loc='upper left')
    plt.show()
    plt.savefig(f'plot_{metric}.png')

if __name__ == '__main__':
    set_memory_growth()
    strategy = auto_select_accelerator()