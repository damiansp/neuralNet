class WideAndDeepModel(Model):
    def __init__(self, units=30, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.dense1 = Dense(units, activation=activation)
        self.dense2 = Dense(units, activation=activation)
        self.main_out = Dense(1)
        self.aux_out = Dense(1)

    def call(self, inputs):
        in_A, in_B = inputs
        dense1 = self.dense1(in_B)
        dense2 = self.dense2(dense1)
        concat = concatenate([in_A, dense2])
        main_out = self.main_out(concat)
        aux_out = self.aux_out(dense2)
        return main_out, aux_out


# inB -> dense1 -> dense2 -> aux_out
#                        \
#                         concat -> main_out
#                        /
#                     inA
