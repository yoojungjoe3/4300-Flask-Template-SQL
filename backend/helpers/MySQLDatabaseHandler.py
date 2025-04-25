import os
import sqlalchemy as db
from sqlalchemy import text

class MySQLDatabaseHandler(object):
    
    IS_DOCKER = True if 'DB_NAME' in os.environ else False

    def __init__(self,MYSQL_USER,MYSQL_USER_PASSWORD,MYSQL_PORT,MYSQL_DATABASE,MYSQL_HOST = "localhost"):
        
        self.MYSQL_HOST = os.environ['DB_NAME'] if MySQLDatabaseHandler.IS_DOCKER else MYSQL_HOST
        self.MYSQL_USER = "admin" if MySQLDatabaseHandler.IS_DOCKER else MYSQL_USER
        self.MYSQL_USER_PASSWORD = "admin" if MySQLDatabaseHandler.IS_DOCKER else MYSQL_USER_PASSWORD
        self.MYSQL_PORT = 3306 if MySQLDatabaseHandler.IS_DOCKER else MYSQL_PORT
        self.MYSQL_DATABASE = "kardashiandb" if MySQLDatabaseHandler.IS_DOCKER else MYSQL_DATABASE
        self.connection = self.validate_connection()
        self.engine = self.connection

    def validate_connection(self):
        print(f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_USER_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}")
        return db.create_engine(f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_USER_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}")

    def lease_connection(self):
        return self.engine.connect()
    
    def query_executor(self,query):
        conn = self.lease_connection()
        if type(query) == list:
            for i in query:
                conn.execute(i)
        else:
            conn.execute(query)
        

    def query_selector(self,query):
        conn = self.lease_connection()
        data = conn.execute(query)
        return data
    
    def query_modifier(self, query, params=None):
        connection = self.lease_connection()
        trans = connection.begin()
        try:
            if params: 
                connection.execute(db.text(query), params)
            else:
                connection.execute(db.text(query))
            trans.commit()
        except Exception as e:
            trans.rollback()
            raise e
        finally:
            connection.close()

    def increment_rating(self, name: str) -> int:
        with self.engine.begin() as conn:
            conn.execute(
                text("UPDATE fics SET Rating = Rating + 1 "
                    "WHERE Name = :name"),
                {"name": name},
            )
            new_rating = conn.scalar(
                text("SELECT Rating FROM fics WHERE Name = :name"),
                {"name": name},
            )
        return new_rating

    def decrement_rating(self, name: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text("UPDATE fics SET Rating = Rating - 1 "
                    "WHERE Name = :name"),
                {"name": name},
            )
            new_rating = conn.scalar(
                text("SELECT Rating FROM fics WHERE Name = :name"),
                {"name": name},
            )
        return new_rating
    
    def get_rating(self, name: str) -> int:
        """Return current Rating for a fic."""
        with self.engine.connect() as conn:
            return conn.scalar(
                text("SELECT Rating FROM fics WHERE Name = :name"),
                {"name": name},
            )
    
    def execute_query(self, query, params=None):
        with self.engine.connect() as connection:
            connection.execute(text(query), params)

    def load_file_into_db(self,file_path = None):
        if MySQLDatabaseHandler.IS_DOCKER:
            return
        if file_path is None:
            file_path = os.path.join(os.environ['ROOT_PATH'], 'init.sql')
        # Drop existing fics table to force reinitialization
        drop_statement = "DROP TABLE IF EXISTS fics;"
        self.query_executor([drop_statement])
        
        # Open and read the SQL file; adjust splitting as needed
        with open(file_path, "r") as sql_file:
            file_content = sql_file.read()
            sql_statements = list(filter(lambda x: x.strip() != '', file_content.split(";\n")))
        
        self.query_executor(sql_statements)