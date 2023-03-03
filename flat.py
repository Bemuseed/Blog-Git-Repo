import numpy

def flatten(matrix):
    shapes = [a.shape for a in matrix]
    offsets = [a.size for a in matrix]
    offsets = numpy.cumsum([0] + offsets)
    result = numpy.concatenate([a.flat for a in matrix])
    return result, shapes, offsets

def unflatten(flattened, shapes, offsets):
    restored = numpy.array([numpy.reshape(flattened[offsets[i]:offsets[i + 1]], shape) for i, shape in enumerate(shapes)])
    return restored

def list_flatten(matrix_list):
    result = numpy.array([], dtype=numpy.float16)
    shapes = []
    offsets = []
    limits = [0]

    counter = 0
    for a in matrix_list:
        counter += 1
        print("Tensor no. " + str(counter))
        r, s, o = flatten(a)
        result = numpy.append(result, r)
        shapes.append(s)
        offsets.append(o)
        limits.append(len(result))
    return result, shapes, offsets, limits

def list_unflatten(flattened, shapes, offsets, limits):
    unflattened = []
    for i in range(len(limits) - 1):
        print("Tensor no. " + str(i+1))
        section = flattened[limits[i]:limits[i+1]]
        unflattened.append(unflatten(section, shapes[i], offsets[i]))
    return unflattened
        

