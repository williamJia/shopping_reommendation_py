from main.app.util.data_engine import data_engine
from main.app.util.date import get_date

class shopping_data:

    def get_shopping_ui_score(self):
        yestoday = get_date.get_yestoday()
        yestoday = '2020-07-23' # todo delete
        sql = """
          select mac as mac,goodid as goodid,actiond as action from test.shopping_ui_score where dt='%s' and actiond <> '0'
        """%yestoday
        df = data_engine.hive2dataframe(sql)
        return df

if __name__ == '__main__':
    sd = shopping_data()
    sd.get_shopping_ui_score()