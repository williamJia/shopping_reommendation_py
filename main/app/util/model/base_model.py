from abc import ABCMeta,abstractmethod
from collections import defaultdict
import pandas as pd

class base_model(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def train(self,df,model_path=''):
        pass

    @abstractmethod
    def predict(self, df, model_path, top_n=10):
        pass

    @abstractmethod
    def fit_transform(self,df,top_k,model_path='temp_index_model.index'):
        pass

    @abstractmethod
    def get_test_data(self):
        pass

    def _change_ret_dict2df(self,dict_rec):
        '''
        变更数据格式，将 {key1:[(rec1,val1),(rec2,val2)],key2:[......], ........} 转化为dataframe 格式，包含 key 和 rec 值
        :param dict_rec: {key1:[(rec1,val1),(rec2,val2)],key2:[......], ........}
        :return: 包含两个字段，id & rec
        '''
        ret_arr = []
        for key in dict_rec:
            row_rec = []
            for t in dict_rec[key]:
                row_rec.append(t[0])
            row_rec_str = ','.join(row_rec)
            ret_arr.append([key,row_rec_str])
        df = pd.DataFrame(ret_arr,columns=['id','rec'])
        return df

    def _get_top_n(self, predictions, n=10):
        '''Return the top-N recommendation for each user from a set of predictions.

        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each user. Default
                is 10.

        Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
        '''

        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n
