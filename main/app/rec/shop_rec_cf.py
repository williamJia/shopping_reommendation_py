import sys
sys.path += ['/data/data_big/jyp/shopping_reommendation_py']

from main.app.dataProcess.shopping_ui_score import shopping_data
from main.app.util.model.cf_knn_basic import cf_knn_basic

class shop_rec:

    def __init__(self,ret_path='shop_rec_cf_item.csv',topN=50,model_path='model.index',user_based=False):
        '''
        多路召回算法
        :param ret_path: 返回结果储存路径
        :param topN: 相似 topN 的数量
        :param model_path: 模型地址
        :param user_based: 协同过滤算法中，是否以 userBase 来计算，默认否，基于itemBase计算
        '''
        self.ret_path = ret_path
        self.topN = topN
        self.model_path = model_path
        self.user_based = user_based

    def shop_rec_cf(self):
        ''' 协同过滤算法 '''
        # 获取数据
        sd = shopping_data()
        df = sd.get_shopping_ui_score()
        df = df.sort_values(by = 'mac',axis = 0,ascending = True) # 相同用户的操作行为可以放在一个分区内，最大程度的保证了预测时不被分离
        print("get data over")

        # 模型训练&预测结果
        ckb = cf_knn_basic(user_based=self.user_based)
        ret = ckb.fit_transform(df,self.topN,model_path=self.model_path,sub_num=10240)
        ret_df = ret.groupby('id',as_index=False).first()
        ret_df.to_csv(self.ret_path,index=False,chunksize=10240)

        print("save data & shop_rec_cf_item_base over")

if __name__ == '__main__':
    # 模型初始化
    sr = shop_rec(ret_path='rec_shop_cf_item.csv',model_path='item_base.index',user_based=False) # 基于 item 的协同过滤
    # sr = shop_rec(ret_path='rec_shop_cf_user.csv',model_path='user_base.index',user_based=True)  # 基于 user 的协同过滤 --》占用内存巨大，pass
    sr.shop_rec_cf()


