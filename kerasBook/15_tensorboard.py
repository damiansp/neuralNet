from keras.callbacks import TensorBoard

TensorBoard(
    log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

# tensorboard --logdir=path_to_logs
