# storage.py
# 哈希表的存储格式


def storage(storage_config, index):
    '''提供storage的配置文件和索引，生成对应存储格式的哈希表
    '''
    if 'dict' in storage_config:
        return InmemoryStorage(storage_config['dict'])
    else:
        raise ValueError("Only in-memory dictionary is supported.")


# 定义抽象类
class BaseStorage(object):
    def __init__(self, config):
        '''使用config进行初始化'''
        raise NotImplementedError

    def keys(self):
        '''返回二进制哈希键'''
        raise NotImplementedError

    def set_val(self, key, val):
        '''设置key处的值val'''
        raise NotImplementedError

    def get_val(self, key):
        '''返回key处的值val'''

    def append_val(self, key, val):
        '''向key处的List中追加值，key不存在则生成Key'''
        raise NotImplementedError

    def get_list(self, key):
        '''返回key处的整个List'''
        raise NotImplementedError

# dict的hash table存储结构
class InmemoryStorage(BaseStorage):
    def __init__(self, config):
        self.name = 'dict'  # 采用字典结构
        self.storage = dict()

    def keys(self):
        return self.storage.keys()

    def set_val(self, key, val):
        self.storage[key] = val

    def get_val(self, key):
        return self.storage[key]

    def append_val(self, key, val):
        self.storage.setdefault(key, []).append(val)

    def get_list(self, key):
        return self.storage.get(key, [])  # 没有key则返回空list
