import surprise
from collections import defaultdict
from surprise import KNNBasic
import pandas as pd
from surprise import Reader
from surprise import Dataset

from main.app.util.model.base_model import base_model


class cf_knn_basic(base_model):

    def __init__(self, rating_scale=(1, 5),k=300, min_k=10, sim_name='cosine', user_based=False):
        '''
        协同过滤模型，基于 knn 计算出相近的值召回
        :param rating_scale: 数据评分范围，默认 1-5 分之间
        :param k: knn 聚类数量
        :param min_k: 最小类别数量
        :param sim_name: 距离计算方法，默认使用余弦相似度
        :param user_based: 默认iteamBase 计算协同过滤，如 user_based，此处改为 true
        '''
        super().__init__()
        self.reader = Reader(rating_scale=rating_scale)
        self.k = k
        self.min_k = min_k
        self.sim_name = sim_name
        self.user_based = user_based

    def train(self, df, model_path=''):
        '''
        协同过滤模型训练
        :param df: 格式包含该三列 --》 userid,iteamid,rating
        :param k: 聚类得类别数量
        :param min_k: 最小聚类数量
        :param sim_name:相似度量指标，默认余弦相似度
        :param user_based:协同过滤基准，默认 itemBase 的协同过滤
        :param model_path:模型持久化地址，默认为空，不执行持久化
        :return: 训练好的模型
        '''
        # 数据类型转换为 surprise 需要的格式
        data = Dataset.load_from_df(df, self.reader)
        trainset = data.build_full_trainset()

        # itemBase 的协同过滤KNN模型的训练和持久化
        algo_knnbasic = KNNBasic(k=self.k, min_k=self.min_k, sim_options={'name': self.sim_name, 'user_based': self.user_based}, verbose=True)
        algo_knnbasic.fit(trainset)
        if model_path:surprise.dump.dump(model_path, algo=algo_knnbasic, verbose=1)

        return algo_knnbasic

    def predict(self, df, model_path, top_n=10):
        '''
        输入待预测dataframe，返回结果集合
        :param df: 待预测数据
        :param model_path: 模型地址
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
        df = pd.DataFrame({'uid': ['u1', 'u2', 'u3', 'u4', 'u5', 'u3', 'u4', 'u5', 'u1', 'u2', 'u3', 'u4', 'u5', 'u3', 'u4', 'u5', 'u1', 'u2', 'u3', 'u4', 'u5', 'u3', 'u4', 'u5'],
                           'vid': ['v1', 'v2', 'v3', 'v1', 'v2', 'v3', 'v1', 'v2', 'v1', 'v2', 'v3', 'v1', 'v2', 'v3','v1', 'v2', 'v1', 'v2', 'v3', 'v1', 'v2', 'v3', 'v1', 'v2'],
                           'rating': [1, 2, 5, 4, 2, 2, 1, 5, 1, 2, 5, 4, 2, 2, 1, 5, 1, 2, 5, 4, 2, 2, 1, 5]})
        return df

if __name__ == '__main__':
    kbc = cf_knn_basic()
    df = kbc.get_test_data()
    # kbc.train(df,model_path='demo_model.pick')
    # ret = kbc.predict(df,'demo_model.pick',3)
    ret = kbc.fit_transform(df,10)
    print(ret)
