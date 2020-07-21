import numpy as np
import faiss
import pandas as pd
import surprise
from collections import defaultdict
from surprise import Reader
from surprise import Dataset
from surprise.prediction_algorithms.matrix_factorization import SVDpp

from main.app.util.model.base_model import base_model

class lfm_svd(base_model):

    def __init__(self, rating_scale=(1, 5)):
        '''
        隐语义模型，基于 矩阵分解 计算出相近的值召回
        :param rating_scale: 数据评分范围，默认 1-5 分之间
        '''
        super().__init__()
        self.reader = Reader(rating_scale=rating_scale)

    def train(self, df,model_path=''):
        '''
        隐语义模型训练
        :param df: 格式包含该三列 --》 userid,iteamid,rating
        :param model_path:模型持久化地址，默认为空，不执行持久化
        :return: 训练好的模型
        '''
        # 数据类型转换为 surprise 需要的格式
        data = Dataset.load_from_df(df, self.reader)
        trainset = data.build_full_trainset()

        algo_lfm = SVDpp()
        algo_lfm.fit(trainset)
        if model_path:surprise.dump.dump(model_path, algo=algo_lfm,verbose=1)
        return algo_lfm

    def predict(self, df, model_path, top_n=10):
        '''
        输入待预测dataframe，返回结果集合
        :param df: 待预测数据
        :param model_path: 模型地址
        :param top_n: 返回值 topn 的数量
        :return: 预测结果集合
        '''
        data = Dataset.load_from_df(df, self.reader)
        trainset = data.build_full_trainset()
        predict_set = trainset.build_anti_testset()

        _, algo = surprise.dump.load(model_path)
        predictions = algo.test(predict_set)
        ret = self._get_top_n(predictions, n=top_n)
        ret_df = self._change_ret_dict2df(ret)
        return ret_df

    def fit_transform(self,df,top_k,model_path='temp_index_model.index'):
        self.train(df,model_path)
        ret = self.predict(df,model_path,top_k)
        return ret

    def get_test_data(self):
        df = pd.DataFrame({'uid': ['u1', 'u2', 'u3', 'u4', 'u5', 'u3', 'u4', 'u5', 'u1', 'u2', 'u3', 'u4', 'u5', 'u3','u4', 'u5', 'u1', 'u2', 'u3', 'u4', 'u5', 'u3', 'u4', 'u5'],
                           'vid': ['v1', 'v2', 'v3', 'v1', 'v2', 'v3', 'v1', 'v2', 'v1', 'v2', 'v3', 'v1', 'v2', 'v3','v1', 'v2', 'v1', 'v2', 'v3', 'v1', 'v2', 'v3', 'v1', 'v2'],
                           'rating': [1, 2, 5, 4, 2, 2, 1, 5, 1, 2, 5, 4, 2, 2, 1, 5, 1, 2, 5, 4, 2, 2, 1, 5]})
        return df

if __name__ == '__main__':
    ls = lfm_svd((1,5))
    # 获取数据，训练 & 持久化 & 预测
    df = ls.get_test_data()
    # index = ls.train(df,'test_lfm_model.picke')
    # ret = ls.predict(df,'test_lfm_model.picke')
    ret = ls.fit_transform(df, 3,'test_lfm_model.picke') # 同时执行训练以及预测召回
    print(ret)
