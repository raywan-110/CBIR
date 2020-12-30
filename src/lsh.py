# lsh.py
# LSH 局部敏感哈希类
import os
import json
import numpy as np
from storage import storage  # import写好的storage

try:
    from bitarray import bitarray
except ImportError:
    bitarray = None

class LSHash(object):
    '''
    LSHash类实现了采用随机投影(random projection)将输入的特征向量映射到哈希字符串所在空间.

    Attributes:
    :param hash_size:
        hashKey的长度. E.g., 32表示32bit hashKey.
    :param input_dim:
        输入特征向量的维度.
    :param num_hashtables:
        (optional) 哈希表的个数,默认为1.
    :param storage_config:
        (optional) 字典结构，形式为{backend_name:config},'backend'在这里采用'dict'.
    :param matrices_filename:
        (optional) 指定存储压缩后后缀为'.npz'的numpy文件的路径，用来存储随机投影矩阵(uniform random planes)
        若不给出具体地址则不存储产生的随机投影矩阵
    :param overwrite:
        (optional) 覆盖已存在的随机矩阵
    '''
    def __init__(self,
                 hash_size,
                 input_dim,
                 num_hashtables=1,
                 storage_config=None,
                 matrices_filename=None,
                 overwrite=False):

        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_hashtables = num_hashtables

        if storage_config is None:
            storage_config = {'dict': None}
        self.storage_config = storage_config

        if matrices_filename and not matrices_filename.endswith('.npz'):
            raise ValueError("The specified file name must end with .npz")
        self.matrices_filename = matrices_filename
        self.overwrite = overwrite
        # 初始化随机投影矩阵
        self._init_uniform_planes()
        # 初始化哈希表
        self._init_hashtables()

    def _init_uniform_planes(self):
        # 检查是否已经存在uniform_planes属性
        if "uniform_planes" in self.__dict__:
            return  # 类的静态函数、类函数、普通函数、全局变量以及一些内置的属性都是放在类__dict__中
        if self.matrices_filename:
            file_exist = os.path.isfile(self.matrices_filename)  # 要传入绝对路径
            if file_exist and not self.overwrite:
                try:
                    npzfiles = np.load(self.matrices_filename)  # 不覆写，读入写好的随机投影矩阵文件
                except IOError:
                    print("Cannot load specified file as a numpy array")
                    raise
                else:
                    npzfiles = sorted(npzfiles.items(), key=lambda x: x[0])  # 压缩存储的时候可能打乱了顺序,这里按照hashtable顺序排序
                    self.uniform_planes = [t[1] for t in npzfiles]
            else:
                # 需要覆写
                self.uniform_planes = [self._generate_uniform_planes for _ in range(self.num_hashtables)]
                try:
                    np.savez_compressed(self.matrices_filename, *self.uniform_planes)
                except IOError:
                    print("IOError when saving matrices to specified path")
                    raise
        else:
            # 不存储产生的随机投影矩阵
            self.uniform_planes = [self._generate_uniform_planes() for _ in range(self.num_hashtables)]

    def _generate_uniform_planes(self):
        '''生成随即投影矩阵，维度为hash_size*input_dim'''
        return np.random.randn(self.hash_size, self.input_dim)

    def _init_hashtables(self):
        self.hash_tables = [storage(self.storage_config, i) for i in range(self.num_hashtables)]

    def _hash(self, planes, input_point):
        '''为输入的一个特征向量生成哈希索引
            :param planes:
                随机投影矩阵，维度为hash_size*input_dim
            :param input_point:
                tuple或者list组织的数据结构，维度为1*input_dim        
        '''
        try:
            input_point = np.array(input_point)
            projections = np.dot(planes, input_point)  # 更快的计算
        except TypeError:
            print("The input point features to be an array-like object with number only elements")
            raise
        except ValueError as e:
            print("""The input_point need to be of the same dimension as
                  'input_dim' when initializing this LSHash instance""", e)
            raise
        else:
            return "".join(['1' if i > 0 else '0' for i in projections])  # 返回哈希字符串存储在hashtable中

    # 将json序列化的数据或者元组存储的数据恢复成为numpy对象
    def _as_np_array(self, tuple_value):
        """ 将以元组形式存储的特征向量进行恢复成为nump array对象
        """
        # 如果存在extra_data, 则第一个元素是tuple类型(安全保存的特征向量)
        # (point:tuple, extra_data). Otherwise (i.e., extra_data=None),
        # return the point stored as a tuple
        tuples = tuple_value

        if isinstance(tuples[0], tuple):
            # 这种情况下tuple的第一个元素是被转换成为tuple的features,后面是extra data
            return np.asarray(tuples[0])  # 转化成为可以计算的数组

        elif isinstance(tuples, (tuple, list)):
            try:
                return np.asarray(tuples)
            except ValueError as e:
                print("The input needs to be an array-like object", e)
                raise
        else:
            raise TypeError("query data is not supported")

    def index(self, input_point, extra_data=None):
        '''为输入的特征向量生成索引并将其添加到hashtable之中
            :param input_point:
                列表元组或者是numpy ndarray对象，维度是1*'input_dim'.这个对象会被转化成为pyhton的元组对象，安全
                储存在hashtable之中。
            :param extra_data:
                (optional)必须是JSON_serializable object: list, dict或者基础数据类型
        '''
        if isinstance(input_point, np.ndarray):
            input_point = input_point.tolist()
        if extra_data:
            value = (tuple(input_point), extra_data)  # extra_data可以包含标签和图像地址
        else:
            value = tuple(input_point)  # 此时value只保存输入的特征向量

        # 保存在hashtable中的格式：{hashkey: [value1,value2,value3...]}
        for i, table in enumerate(self.hash_tables):
            table.append_val(self._hash(self.uniform_planes[i],input_point), value)

    # 根据距离函数返回query_features对应的结果(可以是个数也可以是排序后的结果)
    def query(self, query_point, img_addr, num_results=None, distance_func=None):
        """ 接受一个查询的向量， 根据距离计算函数返回匹配的结果
        :param query_point:
            A list, or tuple, or numpy ndarray, 维度必须是 1 * `input_dim`.
            Used by :meth:`._hash`.
        :param num_results:
            (optional)指定最大的检索数量. 如果没有指定或者num_results小于candidates的大小
            则将所有匹配的candidates排序后返回.
        :param distance_func:
            (optional) 计算匹配度的距离向量("hamming", "euclidean", "true_euclidean",
            "centred_euclidean", "cosine", "l1norm"). By default "euclidean"
            will used.
        """

        candidates = set()  # 忽略多个hashtable中出现的重复元素
        if not distance_func:
            distance_func = "euclidean"  # 默认的距离向量是欧氏距离

        if distance_func == "hamming":
            if not bitarray:
                raise ImportError(" Bitarray is required for hamming distance")

            for i, table in enumerate(self.hash_tables):
                binary_hash = self._hash(self.uniform_planes[i], query_point)
                for key in table.keys():
                    distance = LSHash.hamming_dist(key, binary_hash)  # 计算每个Key与query哈希后的汉明距离
                    if distance < 2:
                        candidates.update(table.get_list(key))  # 将命中的hashtabel中的一栏加入candidate

            d_func = LSHash.euclidean_dist_square  # 设置每一栏的距离计算向量，进行线性查找

        else:

            if distance_func == "euclidean":
                d_func = LSHash.euclidean_dist_square
            elif distance_func == "true_euclidean":
                d_func = LSHash.euclidean_dist
            elif distance_func == "centred_euclidean":
                d_func = LSHash.euclidean_dist_centred
            elif distance_func == "cosine":
                d_func = LSHash.cosine_dist
            elif distance_func == "l1norm":
                d_func = LSHash.l1norm_dist
            else:
                raise ValueError("The distance function name is invalid.")

            for i, table in enumerate(self.hash_tables):
                binary_hash = self._hash(self.uniform_planes[i], query_point)
                candidates.update(table.get_list(binary_hash))  # 更新命中的hashtable中的一栏到candidate中

        # 用距离向量对condidate进行排序 condidate格式： {value1, value2, value3...} value是安全的tuple类型
        print('retrival condidates length:',len(candidates))
        candidates = [{'dis':d_func(query_point, self._as_np_array(ix)),"img":ix[1][0],'cls':ix[1][1]}  
                      for ix in candidates if img_addr != ix[1][0]]
        candidates = sorted(candidates, key=lambda x: x['dis'])  # 用距离进行排序

        return candidates[:num_results] if num_results and num_results < len(candidates) else candidates

    # distance functions
    #定义类方法
    @staticmethod 
    def hamming_dist(bitarray1, bitarray2):
        xor_result = bitarray(bitarray1) ^ bitarray(bitarray2)
        return xor_result.count()

    @staticmethod
    def euclidean_dist(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.array(x) - y
        return np.sqrt(np.dot(diff, diff))

    @staticmethod
    def euclidean_dist_square(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.array(x) - y
        return np.dot(diff, diff)  # (x-y)^2

    @staticmethod
    def euclidean_dist_centred(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.mean(x) - np.mean(y)
        return np.dot(diff, diff)

    @staticmethod
    def l1norm_dist(x, y):
        return sum(abs(x - y))  # L1距离(绝对值之差)

    @staticmethod
    def cosine_dist(x, y):
        return 1 - float(np.dot(x, y)) / ((np.dot(x, x) * np.dot(y, y))**0.5)



