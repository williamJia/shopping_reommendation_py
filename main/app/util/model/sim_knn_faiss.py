import numpy as np
import faiss
from main.app.util.data_engine import data_engine


class sim_knn_faiss:

    def __init__(self,d,m ,nlist = 64, k = 50):
        self.d = d # 原始向量的维度数量
        self.m = m  # 将原始向量切分为 m 个子向量，所以m必须能被d整除
        self.nlist = nlist  # 每个子向量集合都聚成 nlist 类
        self.k = k  # 取topK相似向量
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8) # 默认每个子向量从32bits编码为8bits，如果希望加大压缩比率，也可以编码为4bits

    def train(self,df,ids,model_path = ''):
        '''
        模型训练
        :param df: 输入待训练数据，dataframe类型
        :param ids:输入数据对应的id
        :param model_path:训练好的模型持久化地址，默认空不持久化
        :return:训练好的模型
        '''
        # 转化成 faiss 所需的数据格式 nmarray
        df = df.astype(np.float32)
        data = np.ascontiguousarray(np.array(df))  # 转换成连续类型的数值
        ids = np.array(ids)

        self.index.train(data)
        self.index.add_with_ids(data, ids)  # 添加待id类型的数据，对应输出都是id
        if model_path:faiss.write_index(self.index, model_path) # 持久话索引
        return self.index

    def predict(self,df_ori,k,model_path):
        '''
        预测召回top_K
        :param df:待预测的数据
        :param k: 返回的 topN 的数量
        :param model_path: 模型地址
        :return: 召回结果
        '''
        df = df_ori.astype(np.float32)
        data = np.ascontiguousarray(np.array(df))
        self.index = faiss.read_index(model_path)
        dis, ind = self.index.search(data, k)
        return ind

    def fit_transform(self,df,df_ids,top_k,model_path='temp_index_model.index'):
        self.train(df,df_ids,model_path)
        ret = self.predict(df,top_k,model_path)
        return ret

    def get_test_data(self):
        sql = """
                select goods_id as goods_id,
                    goods_weight as goods_weight,
                    market_price as market_price,
                    shop_price as shop_price,
                    integral as integral,
                    sell_number as sell_number,
                    is_real as is_real,
                    is_alone_sale as is_alone_sale,
                    is_shipping as is_shipping,
                    is_delete as is_delete,
                    is_best as is_best,
                    is_new as is_new,
                    is_hot as is_hot,
                    sell_top as sell_top,
                    is_promote as is_promote,
                    start_sale as start_sale,
                    is_wap as is_wap,
                    isshow as isshow,
                    is_real_subscribe as is_real_subscribe
                FROM coocaa_rds.rds_goods_dim where dt='%s' and is_on_sale=1 and is_delete = 0
        """%"2020-07-16"
        df = data_engine.hive2dataframe(sql)
        df.fillna(-1,inplace=True)
        df_id = df['goods_id']
        df_data = df[[i for i in df.columns.values.tolist() if i not in ['goods_id', 'dt']]]
        return df_data,df_id

if __name__ == '__main__':
    kfs = sim_knn_faiss(18, 3)
    # 获取数据，训练 & 持久化 & 预测
    df_data, df_ids = kfs.get_test_data()
    # index = kfs.train(df_data, df_ids,'test_faiss_model.index')
    # ret = kfs.predict(df_data,5,'test_faiss_model.index')
    ret = kfs.fit_transform(df_data, df_ids, 3) # 同时执行训练以及预测召回
    print(ret)
