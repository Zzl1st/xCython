from tensorflow.examples.tutorials.mnist import input_data

#载入MNIST数据集，如果指定路径下没有已经下好的数据，TF会自动下载

mnist = input_data.read_data_sets("/home/xu/文件/TensorFlow学习笔记/源码/tensorflow-tutorial/Deep_Learning_with_TensorFlow/datasets",one_hot=True)

#打印训练数据集大小
print "Training data size:",mnist.train.num_examples
#打印验证数据集大小
print "Validating data size:",mnist.validation.num_examples
#打印测试数据集大小
print "Testing data size:",mnist.test.num_examples

#打印样本训练数据中的第一个
print "Example training data:",mnist.train.images[0]
#打印样本训练数据的标签的第一个
print "Example training data label:",mnist.train.labels[0]

batch_size=100
xs,ys = mnist.train.next_batch(batch_size)

#从train的集合中选取batch_size个元素
print "X shape:",xs.shape
print "Y shape:",ys.shape


