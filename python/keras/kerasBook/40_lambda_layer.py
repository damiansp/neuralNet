# ex: element-wise squaring of tensor's values:
model.add(lambda(lambda x: x ** 2))

# Compute element-wise euclidean distances between tensors
def euclidean(vecs):
    x, y = vecs
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def euclidean_output_shape(shapes):
    s1, s2 = shapes
    return (s1[0], 1)


lhs_input = Input(shape=[VECTOR_SIZE,])
rhs_input = Input(shape=[VECTOR_SIZE,])
lhs = dense(
    1024, kernel_initialization='glorot_uniform', activation='relu')(lhs_input)
rhs = dense(
    1024, kernel_initialization='glorot_uniform', activation='relu')(lhs_input)
sim = lambda(euclidean, output_shape=euclidean_output_shape([lhs, rhs])
