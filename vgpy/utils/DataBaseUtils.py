import pymssql
from abc import ABCMeta, abstractmethod

from vgpy.utils.logger import create_logger
logger = create_logger()

# for database any operation
class DataBaseUtils(metaclass=ABCMeta):

    # connect DB
    def __init__(self, **kwargs):
        try:
            self.conn = pymssql.connect(
                host = kwargs.get("host"),
                port = kwargs.get("port"),
                user = kwargs.get("user"),
                password = kwargs.get("password"),
                database = kwargs.get("database"),
                charset = "utf8"
            )
            self.cursor = self.conn.cursor(buffered=True)
            logger.info("database connection success")
        except Exception as e:
            print(e)
            logger.info("database connection error")

    # close DB
    def __close(self):
        try:
            self.cursor.close()
            self.conn.close()
            logger.info("DB close success")
        except:
            logger.info("DB close error")

    # update DB
    def update(self, statement, updatedContent):
        print("update", type(updatedContent))
        try:
            if(type(updatedContent) == str):
                print("str", updatedContent)
                self.cursor.execute(statement, (updatedContent))
            elif(type(updatedContent) == tuple):
                print(updatedContent)
                self.cursor.execute(statement, updatedContent)
            self.conn.commit()
            logger.info("DB update success")
            return self.cursor.fetchone()  # return query result
        except Exception as e:
            print(e)
            logger.info("DB update error")
    
    @abstractmethod
    def write_fail_result(self, detectionResult):
        pass

    @abstractmethod
    def write_success_result(self, detectionResult):
        pass

    def write_into_DB(self, detectionResult):
        if(detectionResult == "Fail"):
            self.write_fail_result(detectionResult)
        else: 
            self.write_success_result(detectionResult)
    