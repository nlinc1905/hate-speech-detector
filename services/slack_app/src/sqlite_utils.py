import os
import sqlite3
from sqlite3 import Error


DB_NAME = "hate_speech_tracking.db"


def create_connection(db_file: str):
    """
    Creates a connection to a SQLite DB with the given name.

    :param db_file: name of the DB

    :return: connection object
    """
    return sqlite3.connect(db_file)


def create_table(conn, create_table_sql: str):
    """
    Creates a table in the given SQLite DB.

    :param conn: connection object
    :param create_table_sql: CREATE TABLE command
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def insert_object_into_hate_speech_table(conn, values_to_insert: tuple):
    """
    Insert values into hate_speech table

    :param values_to_insert: e.g. ('ER09JF', 'username', 'i hate u', 166458.02, '3R9FUDJI')
    """
    sql = (
        "INSERT INTO hate_speech (id, user, text, timestamp, channel) "
        "VALUES (?, ?, ?, ?, ?) "
    )
    cur = conn.cursor()
    cur.execute(sql, values_to_insert)
    conn.commit()
    return cur.lastrowid


def insert_object_into_edges_table(conn, values_to_insert: tuple):
    """
    Insert values into edges table

    :param values_to_insert: e.g. ('username1', 'username2', 166458.02)
    """
    sql = (
        "INSERT INTO edges (source, target, timestamp) "
        "VALUES (?, ?, ?) "
    )
    cur = conn.cursor()
    cur.execute(sql, values_to_insert)
    conn.commit()
    return cur.lastrowid


def setup_db():
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)

    sql_create_hate_speech_table = (
        "CREATE TABLE IF NOT EXISTS hate_speech ("
            "id text PRIMARY KEY, "
            "user text NOT NULL, "
            "text text NOT NULL, "
            "timestamp numeric NOT NULL, "
            "channel text NOT NULL"
        ");"
    )

    sql_create_edges_table = (
        "CREATE TABLE IF NOT EXISTS edges ("
            "source text NOT NULL, "
            "target text NOT NULL, "
            "timestamp numeric NOT NULL "
        ");"
    )

    conn = create_connection(DB_NAME)
    create_table(conn, sql_create_hate_speech_table)
    create_table(conn, sql_create_edges_table)


def update_db(hate_speech: tuple = None, edges: tuple = None):
    conn = create_connection(DB_NAME)
    with conn:
        if hate_speech is not None:
            insert_object_into_hate_speech_table(conn, hate_speech)
        if edges is not None:
            insert_object_into_edges_table(conn, edges)


if __name__ == '__main__':
    setup_db()
    update_db(hate_speech=('ER09JF', 'username', 'i hate u', 166458.02, '3R9FUDJI'))
    update_db(edges=('username1', 'username2', 166458.02))
    update_db(
        hate_speech=('ER09JG', 'username2', 'i hate u 2', 166458.02, '3R9FUDJI'),
        edges=('username2', 'username1', 166458.02)
    )
