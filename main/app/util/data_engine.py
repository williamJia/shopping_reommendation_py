from impala.dbapi import connect
from impala.util import as_pandas

class data_engine:

    @staticmethod
    def hive2dataframe(sql_context):
        conn = connect(host='192.168.1.73', port=10000, auth_mechanism='PLAIN', user='jiayuepeng', password='wVnxjyp',database='ods')
        cursor = conn.cursor()
        cursor.execute(sql_context)
        return as_pandas(cursor)
