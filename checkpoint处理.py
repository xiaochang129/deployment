# coding=utf-8
#inference by checkpoint without model structure

import tensorflow as tf
grath=tf.Graph()
with graph.as_default():
    #模型结构
    output=tf...
with tf.Session(Graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    saver=tf.train.import_meta_graph(config["model_path"]+'/model.meta')
    saver.restore(sess,config["model_path"]+"/model")
    #模型数据:  model.data   变量数据
    #           model.index
    #           model.meta   保存了graph结构，包括GraphDef：图结构, SaverDef：张量名称等


#save checkpoint
    saver=tf.train.Saver()
    ...
    saver.save(sess,"model_path/checkpoint.ckpt",global_step=step)
#问题: 占内存过多。


#读取pb
    BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # 瓶颈层输出张量名称
    JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # 输入层张量名称
    MODEL_DIR = './inception_dec_2015'  # 模型存放文件夹
    MODEL_FILE = 'tensorflow_inception_graph.pb'  # 模型名
    # 加载模型
    # with gfile.FastGFile(os.path.join(MODEL_DIR,MODEL_FILE),'rb') as f:   # 阅读器上下文
    graph_def = tf.GraphDef()  # 生成图
    graph_def.ParseFromString(open(os.path.join(MODEL_DIR, MODEL_FILE), 'rb').read())  # 图加载模型
    # 加载图上节点张量(按照句柄理解)
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def( graph_def,return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])
