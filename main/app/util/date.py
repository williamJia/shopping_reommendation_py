import datetime

class get_date:

    @staticmethod
    def get_yestoday():
        yesterday = (datetime.datetime.today() + datetime.timedelta(days = -1)).strftime("%Y-%m-%d")    # 昨天日期
        return yesterday
