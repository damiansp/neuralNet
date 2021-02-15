class Callback:
    def __init__(self):
        self.validation_data = None
        self.model = None

    def on_epoch_begin(self, epoch, logs=None):
        '''Called at beginning of training epoch'''
        pass

    def on_epoch_end(self, epoch, logs=None):
        '''Called at end of training epoch'''
        pass

    # def on_{train|test|predict}_{begin|end}(self, logs=None): ...
