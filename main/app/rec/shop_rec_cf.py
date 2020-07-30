import sys
sys.path += ['/data/data_big/jyp/shopping_reommendation_py']

from main.app.dataProcess.shopping_ui_score import shopping_data
from main.app.util.model.cf_knn_basic import cf_knn_basic

class shop_rec_cf:

    def shop_rec_cf_item_base(self,ret_path='shop_rec_cf.csv',topN=50,model_path='item_base.index'):
        ''' 基于 itemBase 的协同过滤算法 '''
        # 获取数据
        sd = shopping_data()
        df = sd.get_shopping_ui_score()
        df = df.sort_values(by = 'mac',axis = 0,ascending = True) # 相同用户的操作行为可以放在一个分区内，最大程度的保证了预测时不被分离
        print("get data over")

        # 模型训练&预测结果
        ckb = cf_knn_basic()
        print("model train begin")
        ret = ckb.fit_transform(df,topN,model_path=model_path,sub_num=10240)
        ret_df = ret.groupby('id',as_index=False).first()
        print("model over")
        ret_df.to_csv(ret_path,index=False,chunksize=10240)

        print("save data & shop_rec_cf_item_base over")

if __name__ == '__main__':
    sr = shop_rec_cf()
    sr.shop_rec_cf_item_base()