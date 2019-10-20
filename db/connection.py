import pymongo

class MongoDbConnection:
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")

    @staticmethod
    def getCollection(db):
        return MongoDbConnection.myclient[db]