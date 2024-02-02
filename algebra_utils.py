import tensorflow as tf
import lattice as lt


def dot(v1: tf.Tensor, v2: tf.Tensor):
    if v1.shape != v2.shape:
        raise RuntimeError("Incompatible vector shapes!")
    if len(v1.shape) == 3:
        return lt.global_sum(tf.einsum("mic,mic", tf.math.conj(v1), v2))
    if len(v1.shape) == 4:
        return lt.global_sum(tf.einsum("mijc,mijc", tf.math.conj(v1), v2))
