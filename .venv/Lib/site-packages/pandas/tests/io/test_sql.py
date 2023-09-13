"""SQL io tests

The SQL tests are broken down in different classes:

- `PandasSQLTest`: base class with common methods for all test classes
- Tests for the public API (only tests with sqlite3)
    - `_TestSQLApi` base class
    - `TestSQLApi`: test the public API with sqlalchemy engine
    - `TestSQLiteFallbackApi`: test the public API with a sqlite DBAPI
      connection
- Tests for the different SQL flavors (flavor specific type conversions)
    - Tests for the sqlalchemy mode: `_TestSQLAlchemy` is the base class with
      common methods. The different tested flavors (sqlite3, MySQL,
      PostgreSQL) derive from the base class
    - Tests for the fallback mode (`TestSQLiteFallback`)

"""
from __future__ import annotations

import contextlib
from contextlib import closing
import csv
from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
from io import StringIO
from pathlib import Path
import sqlite3
import uuid

import numpy as np
import pytest

from pandas._libs import lib
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeTZDtype,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    concat,
    date_range,
    isna,
    to_datetime,
    to_timedelta,
)
import pandas._testing as tm
from pandas.core.arrays import (
    ArrowStringArray,
    StringArray,
)
from pandas.util.version import Version

from pandas.io import sql
from pandas.io.sql import (
    SQLAlchemyEngine,
    SQLDatabase,
    SQLiteDatabase,
    get_engine,
    pandasSQL_builder,
    read_sql_query,
    read_sql_table,
)

try:
    import sqlalchemy

    SQLALCHEMY_INSTALLED = True
except ImportError:
    SQLALCHEMY_INSTALLED = False


@pytest.fixture
def sql_strings():
    return {
        "read_parameters": {
            "sqlite": "SELECT * FROM iris WHERE Name=? AND SepalLength=?",
            "mysql": "SELECT * FROM iris WHERE `Name`=%s AND `SepalLength`=%s",
            "postgresql": 'SELECT * FROM iris WHERE "Name"=%s AND "SepalLength"=%s',
        },
        "read_named_parameters": {
            "sqlite": """
                SELECT * FROM iris WHERE Name=:name AND SepalLength=:length
                """,
            "mysql": """
                SELECT * FROM iris WHERE
                `Name`=%(name)s AND `SepalLength`=%(length)s
                """,
            "postgresql": """
                SELECT * FROM iris WHERE
                "Name"=%(name)s AND "SepalLength"=%(length)s
                """,
        },
        "read_no_parameters_with_percent": {
            "sqlite": "SELECT * FROM iris WHERE Name LIKE '%'",
            "mysql": "SELECT * FROM iris WHERE `Name` LIKE '%'",
            "postgresql": "SELECT * FROM iris WHERE \"Name\" LIKE '%'",
        },
    }


def iris_table_metadata(dialect: str):
    from sqlalchemy import (
        REAL,
        Column,
        Float,
        MetaData,
        String,
        Table,
    )

    dtype = Float if dialect == "postgresql" else REAL
    metadata = MetaData()
    iris = Table(
        "iris",
        metadata,
        Column("SepalLength", dtype),
        Column("SepalWidth", dtype),
        Column("PetalLength", dtype),
        Column("PetalWidth", dtype),
        Column("Name", String(200)),
    )
    return iris


def create_and_load_iris_sqlite3(conn: sqlite3.Connection, iris_file: Path):
    cur = conn.cursor()
    stmt = """CREATE TABLE iris (
            "SepalLength" REAL,
            "SepalWidth" REAL,
            "PetalLength" REAL,
            "PetalWidth" REAL,
            "Name" TEXT
        )"""
    cur.execute(stmt)
    with iris_file.open(newline=None, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        stmt = "INSERT INTO iris VALUES(?, ?, ?, ?, ?)"
        cur.executemany(stmt, reader)


def create_and_load_iris(conn, iris_file: Path, dialect: str):
    from sqlalchemy import insert
    from sqlalchemy.engine import Engine

    iris = iris_table_metadata(dialect)

    with iris_file.open(newline=None, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        params = [dict(zip(header, row)) for row in reader]
        stmt = insert(iris).values(params)
        if isinstance(conn, Engine):
            with conn.connect() as conn:
                with conn.begin():
                    iris.drop(conn, checkfirst=True)
                    iris.create(bind=conn)
                    conn.execute(stmt)
        else:
            with conn.begin():
                iris.drop(conn, checkfirst=True)
                iris.create(bind=conn)
                conn.execute(stmt)


def create_and_load_iris_view(conn):
    stmt = "CREATE VIEW iris_view AS SELECT * FROM iris"
    if isinstance(conn, sqlite3.Connection):
        cur = conn.cursor()
        cur.execute(stmt)
    else:
        from sqlalchemy import text
        from sqlalchemy.engine import Engine

        stmt = text(stmt)
        if isinstance(conn, Engine):
            with conn.connect() as conn:
                with conn.begin():
                    conn.execute(stmt)
        else:
            with conn.begin():
                conn.execute(stmt)


def types_table_metadata(dialect: str):
    from sqlalchemy import (
        TEXT,
        Boolean,
        Column,
        DateTime,
        Float,
        Integer,
        MetaData,
        Table,
    )

    date_type = TEXT if dialect == "sqlite" else DateTime
    bool_type = Integer if dialect == "sqlite" else Boolean
    metadata = MetaData()
    types = Table(
        "types",
        metadata,
        Column("TextCol", TEXT),
        Column("DateCol", date_type),
        Column("IntDateCol", Integer),
        Column("IntDateOnlyCol", Integer),
        Column("FloatCol", Float),
        Column("IntCol", Integer),
        Column("BoolCol", bool_type),
        Column("IntColWithNull", Integer),
        Column("BoolColWithNull", bool_type),
    )
    if dialect == "postgresql":
        types.append_column(Column("DateColWithTz", DateTime(timezone=True)))
    return types


def create_and_load_types_sqlite3(conn: sqlite3.Connection, types_data: list[dict]):
    cur = conn.cursor()
    stmt = """CREATE TABLE types (
                    "TextCol" TEXT,
                    "DateCol" TEXT,
                    "IntDateCol" INTEGER,
                    "IntDateOnlyCol" INTEGER,
                    "FloatCol" REAL,
                    "IntCol" INTEGER,
                    "BoolCol" INTEGER,
                    "IntColWithNull" INTEGER,
                    "BoolColWithNull" INTEGER
                )"""
    cur.execute(stmt)

    stmt = """
            INSERT INTO types
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
    cur.executemany(stmt, types_data)


def create_and_load_types(conn, types_data: list[dict], dialect: str):
    from sqlalchemy import insert
    from sqlalchemy.engine import Engine

    types = types_table_metadata(dialect)

    stmt = insert(types).values(types_data)
    if isinstance(conn, Engine):
        with conn.connect() as conn:
            with conn.begin():
                types.drop(conn, checkfirst=True)
                types.create(bind=conn)
                conn.execute(stmt)
    else:
        with conn.begin():
            types.drop(conn, checkfirst=True)
            types.create(bind=conn)
            conn.execute(stmt)


def check_iris_frame(frame: DataFrame):
    pytype = frame.dtypes.iloc[0].type
    row = frame.iloc[0]
    assert issubclass(pytype, np.floating)
    tm.equalContents(row.values, [5.1, 3.5, 1.4, 0.2, "Iris-setosa"])
    assert frame.shape in ((150, 5), (8, 5))


def count_rows(conn, table_name: str):
    stmt = f"SELECT count(*) AS count_1 FROM {table_name}"
    if isinstance(conn, sqlite3.Connection):
        cur = conn.cursor()
        return cur.execute(stmt).fetchone()[0]
    else:
        from sqlalchemy import create_engine
        from sqlalchemy.engine import Engine

        if isinstance(conn, str):
            try:
                engine = create_engine(conn)
                with engine.connect() as conn:
                    return conn.exec_driver_sql(stmt).scalar_one()
            finally:
                engine.dispose()
        elif isinstance(conn, Engine):
            with conn.connect() as conn:
                return conn.exec_driver_sql(stmt).scalar_one()
        else:
            return conn.exec_driver_sql(stmt).scalar_one()


@pytest.fixture
def iris_path(datapath):
    iris_path = datapath("io", "data", "csv", "iris.csv")
    return Path(iris_path)


@pytest.fixture
def types_data():
    return [
        {
            "TextCol": "first",
            "DateCol": "2000-01-03 00:00:00",
            "IntDateCol": 535852800,
            "IntDateOnlyCol": 20101010,
            "FloatCol": 10.10,
            "IntCol": 1,
            "BoolCol": False,
            "IntColWithNull": 1,
            "BoolColWithNull": False,
            "DateColWithTz": "2000-01-01 00:00:00-08:00",
        },
        {
            "TextCol": "first",
            "DateCol": "2000-01-04 00:00:00",
            "IntDateCol": 1356998400,
            "IntDateOnlyCol": 20101212,
            "FloatCol": 10.10,
            "IntCol": 1,
            "BoolCol": False,
            "IntColWithNull": None,
            "BoolColWithNull": None,
            "DateColWithTz": "2000-06-01 00:00:00-07:00",
        },
    ]


@pytest.fixture
def types_data_frame(types_data):
    dtypes = {
        "TextCol": "str",
        "DateCol": "str",
        "IntDateCol": "int64",
        "IntDateOnlyCol": "int64",
        "FloatCol": "float",
        "IntCol": "int64",
        "BoolCol": "int64",
        "IntColWithNull": "float",
        "BoolColWithNull": "float",
    }
    df = DataFrame(types_data)
    return df[dtypes.keys()].astype(dtypes)


@pytest.fixture
def test_frame1():
    columns = ["index", "A", "B", "C", "D"]
    data = [
        (
            "2000-01-03 00:00:00",
            0.980268513777,
            3.68573087906,
            -0.364216805298,
            -1.15973806169,
        ),
        (
            "2000-01-04 00:00:00",
            1.04791624281,
            -0.0412318367011,
            -0.16181208307,
            0.212549316967,
        ),
        (
            "2000-01-05 00:00:00",
            0.498580885705,
            0.731167677815,
            -0.537677223318,
            1.34627041952,
        ),
        (
            "2000-01-06 00:00:00",
            1.12020151869,
            1.56762092543,
            0.00364077397681,
            0.67525259227,
        ),
    ]
    return DataFrame(data, columns=columns)


@pytest.fixture
def test_frame3():
    columns = ["index", "A", "B"]
    data = [
        ("2000-01-03 00:00:00", 2**31 - 1, -1.987670),
        ("2000-01-04 00:00:00", -29, -0.0412318367011),
        ("2000-01-05 00:00:00", 20000, 0.731167677815),
        ("2000-01-06 00:00:00", -290867, 1.56762092543),
    ]
    return DataFrame(data, columns=columns)


@pytest.fixture
def mysql_pymysql_engine(iris_path, types_data):
    sqlalchemy = pytest.importorskip("sqlalchemy")
    pymysql = pytest.importorskip("pymysql")
    engine = sqlalchemy.create_engine(
        "mysql+pymysql://root@localhost:3306/pandas",
        connect_args={"client_flag": pymysql.constants.CLIENT.MULTI_STATEMENTS},
        poolclass=sqlalchemy.pool.NullPool,
    )
    insp = sqlalchemy.inspect(engine)
    if not insp.has_table("iris"):
        create_and_load_iris(engine, iris_path, "mysql")
    if not insp.has_table("types"):
        for entry in types_data:
            entry.pop("DateColWithTz")
        create_and_load_types(engine, types_data, "mysql")
    yield engine
    with engine.connect() as conn:
        with conn.begin():
            stmt = sqlalchemy.text("DROP TABLE IF EXISTS test_frame;")
            conn.execute(stmt)
    engine.dispose()


@pytest.fixture
def mysql_pymysql_conn(mysql_pymysql_engine):
    with mysql_pymysql_engine.connect() as conn:
        yield conn


@pytest.fixture
def postgresql_psycopg2_engine(iris_path, types_data):
    sqlalchemy = pytest.importorskip("sqlalchemy")
    pytest.importorskip("psycopg2")
    engine = sqlalchemy.create_engine(
        "postgresql+psycopg2://postgres:postgres@localhost:5432/pandas",
        poolclass=sqlalchemy.pool.NullPool,
    )
    insp = sqlalchemy.inspect(engine)
    if not insp.has_table("iris"):
        create_and_load_iris(engine, iris_path, "postgresql")
    if not insp.has_table("types"):
        create_and_load_types(engine, types_data, "postgresql")
    yield engine
    with engine.connect() as conn:
        with conn.begin():
            stmt = sqlalchemy.text("DROP TABLE IF EXISTS test_frame;")
            conn.execute(stmt)
    engine.dispose()


@pytest.fixture
def postgresql_psycopg2_conn(postgresql_psycopg2_engine):
    with postgresql_psycopg2_engine.connect() as conn:
        yield conn


@pytest.fixture
def sqlite_str():
    pytest.importorskip("sqlalchemy")
    with tm.ensure_clean() as name:
        yield "sqlite:///" + name


@pytest.fixture
def sqlite_engine(sqlite_str):
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(sqlite_str, poolclass=sqlalchemy.pool.NullPool)
    yield engine
    engine.dispose()


@pytest.fixture
def sqlite_conn(sqlite_engine):
    with sqlite_engine.connect() as conn:
        yield conn


@pytest.fixture
def sqlite_iris_str(sqlite_str, iris_path):
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(sqlite_str)
    create_and_load_iris(engine, iris_path, "sqlite")
    engine.dispose()
    return sqlite_str


@pytest.fixture
def sqlite_iris_engine(sqlite_engine, iris_path):
    create_and_load_iris(sqlite_engine, iris_path, "sqlite")
    return sqlite_engine


@pytest.fixture
def sqlite_iris_conn(sqlite_iris_engine):
    with sqlite_iris_engine.connect() as conn:
        yield conn


@pytest.fixture
def sqlite_buildin():
    with contextlib.closing(sqlite3.connect(":memory:")) as closing_conn:
        with closing_conn as conn:
            yield conn


@pytest.fixture
def sqlite_buildin_iris(sqlite_buildin, iris_path):
    create_and_load_iris_sqlite3(sqlite_buildin, iris_path)
    return sqlite_buildin


mysql_connectable = [
    "mysql_pymysql_engine",
    "mysql_pymysql_conn",
]


postgresql_connectable = [
    "postgresql_psycopg2_engine",
    "postgresql_psycopg2_conn",
]

sqlite_connectable = [
    "sqlite_engine",
    "sqlite_conn",
    "sqlite_str",
]

sqlite_iris_connectable = [
    "sqlite_iris_engine",
    "sqlite_iris_conn",
    "sqlite_iris_str",
]

sqlalchemy_connectable = mysql_connectable + postgresql_connectable + sqlite_connectable

sqlalchemy_connectable_iris = (
    mysql_connectable + postgresql_connectable + sqlite_iris_connectable
)

all_connectable = sqlalchemy_connectable + ["sqlite_buildin"]

all_connectable_iris = sqlalchemy_connectable_iris + ["sqlite_buildin_iris"]


@pytest.mark.db
@pytest.mark.parametrize("conn", all_connectable)
def test_dataframe_to_sql(conn, test_frame1, request):
    # GH 51086 if conn is sqlite_engine
    conn = request.getfixturevalue(conn)
    test_frame1.to_sql(name="test", con=conn, if_exists="append", index=False)


@pytest.mark.db
@pytest.mark.parametrize("conn", all_connectable)
def test_dataframe_to_sql_arrow_dtypes(conn, request):
    # GH 52046
    pytest.importorskip("pyarrow")
    df = DataFrame(
        {
            "int": pd.array([1], dtype="int8[pyarrow]"),
            "datetime": pd.array(
                [datetime(2023, 1, 1)], dtype="timestamp[ns][pyarrow]"
            ),
            "date": pd.array([date(2023, 1, 1)], dtype="date32[day][pyarrow]"),
            "timedelta": pd.array([timedelta(1)], dtype="duration[ns][pyarrow]"),
            "string": pd.array(["a"], dtype="string[pyarrow]"),
        }
    )
    conn = request.getfixturevalue(conn)
    with tm.assert_produces_warning(UserWarning, match="the 'timedelta'"):
        df.to_sql(name="test_arrow", con=conn, if_exists="replace", index=False)


@pytest.mark.db
@pytest.mark.parametrize("conn", all_connectable)
def test_dataframe_to_sql_arrow_dtypes_missing(conn, request, nulls_fixture):
    # GH 52046
    pytest.importorskip("pyarrow")
    df = DataFrame(
        {
            "datetime": pd.array(
                [datetime(2023, 1, 1), nulls_fixture], dtype="timestamp[ns][pyarrow]"
            ),
        }
    )
    conn = request.getfixturevalue(conn)
    df.to_sql(name="test_arrow", con=conn, if_exists="replace", index=False)


@pytest.mark.db
@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("method", [None, "multi"])
def test_to_sql(conn, method, test_frame1, request):
    conn = request.getfixturevalue(conn)
    with pandasSQL_builder(conn, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, "test_frame", method=method)
        assert pandasSQL.has_table("test_frame")
    assert count_rows(conn, "test_frame") == len(test_frame1)


@pytest.mark.db
@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("mode, num_row_coef", [("replace", 1), ("append", 2)])
def test_to_sql_exist(conn, mode, num_row_coef, test_frame1, request):
    conn = request.getfixturevalue(conn)
    with pandasSQL_builder(conn, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, "test_frame", if_exists="fail")
        pandasSQL.to_sql(test_frame1, "test_frame", if_exists=mode)
        assert pandasSQL.has_table("test_frame")
    assert count_rows(conn, "test_frame") == num_row_coef * len(test_frame1)


@pytest.mark.db
@pytest.mark.parametrize("conn", all_connectable)
def test_to_sql_exist_fail(conn, test_frame1, request):
    conn = request.getfixturevalue(conn)
    with pandasSQL_builder(conn, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, "test_frame", if_exists="fail")
        assert pandasSQL.has_table("test_frame")

        msg = "Table 'test_frame' already exists"
        with pytest.raises(ValueError, match=msg):
            pandasSQL.to_sql(test_frame1, "test_frame", if_exists="fail")


@pytest.mark.db
@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_iris_query(conn, request):
    conn = request.getfixturevalue(conn)
    iris_frame = read_sql_query("SELECT * FROM iris", conn)
    check_iris_frame(iris_frame)
    iris_frame = pd.read_sql("SELECT * FROM iris", conn)
    check_iris_frame(iris_frame)
    iris_frame = pd.read_sql("SELECT * FROM iris where 0=1", conn)
    assert iris_frame.shape == (0, 5)
    assert "SepalWidth" in iris_frame.columns


@pytest.mark.db
@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_iris_query_chunksize(conn, request):
    conn = request.getfixturevalue(conn)
    iris_frame = concat(read_sql_query("SELECT * FROM iris", conn, chunksize=7))
    check_iris_frame(iris_frame)
    iris_frame = concat(pd.read_sql("SELECT * FROM iris", conn, chunksize=7))
    check_iris_frame(iris_frame)
    iris_frame = concat(pd.read_sql("SELECT * FROM iris where 0=1", conn, chunksize=7))
    assert iris_frame.shape == (0, 5)
    assert "SepalWidth" in iris_frame.columns


@pytest.mark.db
@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_read_iris_query_expression_with_parameter(conn, request):
    conn = request.getfixturevalue(conn)
    from sqlalchemy import (
        MetaData,
        Table,
        create_engine,
        select,
    )

    metadata = MetaData()
    autoload_con = create_engine(conn) if isinstance(conn, str) else conn
    iris = Table("iris", metadata, autoload_with=autoload_con)
    iris_frame = read_sql_query(
        select(iris), conn, params={"name": "Iris-setosa", "length": 5.1}
    )
    check_iris_frame(iris_frame)
    if isinstance(conn, str):
        autoload_con.dispose()


@pytest.mark.db
@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_iris_query_string_with_parameter(conn, request, sql_strings):
    for db, query in sql_strings["read_parameters"].items():
        if db in conn:
            break
    else:
        raise KeyError(f"No part of {conn} found in sql_strings['read_parameters']")
    conn = request.getfixturevalue(conn)
    iris_frame = read_sql_query(query, conn, params=("Iris-setosa", 5.1))
    check_iris_frame(iris_frame)


@pytest.mark.db
@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_read_iris_table(conn, request):
    # GH 51015 if conn = sqlite_iris_str
    conn = request.getfixturevalue(conn)
    iris_frame = read_sql_table("iris", conn)
    check_iris_frame(iris_frame)
    iris_frame = pd.read_sql("iris", conn)
    check_iris_frame(iris_frame)


@pytest.mark.db
@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_read_iris_table_chunksize(conn, request):
    conn = request.getfixturevalue(conn)
    iris_frame = concat(read_sql_table("iris", conn, chunksize=7))
    check_iris_frame(iris_frame)
    iris_frame = concat(pd.read_sql("iris", conn, chunksize=7))
    check_iris_frame(iris_frame)


@pytest.mark.db
@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_to_sql_callable(conn, test_frame1, request):
    conn = request.getfixturevalue(conn)

    check = []  # used to double check function below is really being used

    def sample(pd_table, conn, keys, data_iter):
        check.append(1)
        data = [dict(zip(keys, row)) for row in data_iter]
        conn.execute(pd_table.table.insert(), data)

    with pandasSQL_builder(conn, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, "test_frame", method=sample)
        assert pandasSQL.has_table("test_frame")
    assert check == [1]
    assert count_rows(conn, "test_frame") == len(test_frame1)


@pytest.mark.db
@pytest.mark.parametrize("conn", mysql_connectable)
def test_default_type_conversion(conn, request):
    conn = request.getfixturevalue(conn)
    df = sql.read_sql_table("types", conn)

    assert issubclass(df.FloatCol.dtype.type, np.floating)
    assert issubclass(df.IntCol.dtype.type, np.integer)

    # MySQL has no real BOOL type (it's an alias for TINYINT)
    assert issubclass(df.BoolCol.dtype.type, np.integer)

    # Int column with NA values stays as float
    assert issubclass(df.IntColWithNull.dtype.type, np.floating)

    # Bool column with NA = int column with NA values => becomes float
    assert issubclass(df.BoolColWithNull.dtype.type, np.floating)


@pytest.mark.db
@pytest.mark.parametrize("conn", mysql_connectable)
def test_read_procedure(conn, request):
    conn = request.getfixturevalue(conn)

    # GH 7324
    # Although it is more an api test, it is added to the
    # mysql tests as sqlite does not have stored procedures
    from sqlalchemy import text
    from sqlalchemy.engine import Engine

    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    df.to_sql(name="test_frame", con=conn, index=False)

    proc = """DROP PROCEDURE IF EXISTS get_testdb;

    CREATE PROCEDURE get_testdb ()

    BEGIN
        SELECT * FROM test_frame;
    END"""
    proc = text(proc)
    if isinstance(conn, Engine):
        with conn.connect() as engine_conn:
            with engine_conn.begin():
                engine_conn.execute(proc)
    else:
        with conn.begin():
            conn.execute(proc)

    res1 = sql.read_sql_query("CALL get_testdb();", conn)
    tm.assert_frame_equal(df, res1)

    # test delegation to read_sql_query
    res2 = sql.read_sql("CALL get_testdb();", conn)
    tm.assert_frame_equal(df, res2)


@pytest.mark.db
@pytest.mark.parametrize("conn", postgresql_connectable)
@pytest.mark.parametrize("expected_count", [2, "Success!"])
def test_copy_from_callable_insertion_method(conn, expected_count, request):
    # GH 8953
    # Example in io.rst found under _io.sql.method
    # not available in sqlite, mysql
    def psql_insert_copy(table, conn, keys, data_iter):
        # gets a DBAPI connection that can provide a cursor
        dbapi_conn = conn.connection
        with dbapi_conn.cursor() as cur:
            s_buf = StringIO()
            writer = csv.writer(s_buf)
            writer.writerows(data_iter)
            s_buf.seek(0)

            columns = ", ".join([f'"{k}"' for k in keys])
            if table.schema:
                table_name = f"{table.schema}.{table.name}"
            else:
                table_name = table.name

            sql_query = f"COPY {table_name} ({columns}) FROM STDIN WITH CSV"
            cur.copy_expert(sql=sql_query, file=s_buf)
        return expected_count

    conn = request.getfixturevalue(conn)
    expected = DataFrame({"col1": [1, 2], "col2": [0.1, 0.2], "col3": ["a", "n"]})
    result_count = expected.to_sql(
        name="test_frame", con=conn, index=False, method=psql_insert_copy
    )
    # GH 46891
    if expected_count is None:
        assert result_count is None
    else:
        assert result_count == expected_count
    result = sql.read_sql_table("test_frame", conn)
    tm.assert_frame_equal(result, expected)


@pytest.mark.db
@pytest.mark.parametrize("conn", postgresql_connectable)
def test_insertion_method_on_conflict_do_nothing(conn, request):
    # GH 15988: Example in to_sql docstring
    conn = request.getfixturevalue(conn)

    from sqlalchemy.dialects.postgresql import insert
    from sqlalchemy.engine import Engine
    from sqlalchemy.sql import text

    def insert_on_conflict(table, conn, keys, data_iter):
        data = [dict(zip(keys, row)) for row in data_iter]
        stmt = (
            insert(table.table)
            .values(data)
            .on_conflict_do_nothing(index_elements=["a"])
        )
        result = conn.execute(stmt)
        return result.rowcount

    create_sql = text(
        """
    CREATE TABLE test_insert_conflict (
        a  integer PRIMARY KEY,
        b  numeric,
        c  text
    );
    """
    )
    if isinstance(conn, Engine):
        with conn.connect() as con:
            with con.begin():
                con.execute(create_sql)
    else:
        with conn.begin():
            conn.execute(create_sql)

    expected = DataFrame([[1, 2.1, "a"]], columns=list("abc"))
    expected.to_sql(
        name="test_insert_conflict", con=conn, if_exists="append", index=False
    )

    df_insert = DataFrame([[1, 3.2, "b"]], columns=list("abc"))
    inserted = df_insert.to_sql(
        name="test_insert_conflict",
        con=conn,
        index=False,
        if_exists="append",
        method=insert_on_conflict,
    )
    result = sql.read_sql_table("test_insert_conflict", conn)
    tm.assert_frame_equal(result, expected)
    assert inserted == 0

    # Cleanup
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("test_insert_conflict")


@pytest.mark.db
@pytest.mark.parametrize("conn", mysql_connectable)
def test_insertion_method_on_conflict_update(conn, request):
    # GH 14553: Example in to_sql docstring
    conn = request.getfixturevalue(conn)

    from sqlalchemy.dialects.mysql import insert
    from sqlalchemy.engine import Engine
    from sqlalchemy.sql import text

    def insert_on_conflict(table, conn, keys, data_iter):
        data = [dict(zip(keys, row)) for row in data_iter]
        stmt = insert(table.table).values(data)
        stmt = stmt.on_duplicate_key_update(b=stmt.inserted.b, c=stmt.inserted.c)
        result = conn.execute(stmt)
        return result.rowcount

    create_sql = text(
        """
    CREATE TABLE test_insert_conflict (
        a INT PRIMARY KEY,
        b FLOAT,
        c VARCHAR(10)
    );
    """
    )
    if isinstance(conn, Engine):
        with conn.connect() as con:
            with con.begin():
                con.execute(create_sql)
    else:
        with conn.begin():
            conn.execute(create_sql)

    df = DataFrame([[1, 2.1, "a"]], columns=list("abc"))
    df.to_sql(name="test_insert_conflict", con=conn, if_exists="append", index=False)

    expected = DataFrame([[1, 3.2, "b"]], columns=list("abc"))
    inserted = expected.to_sql(
        name="test_insert_conflict",
        con=conn,
        index=False,
        if_exists="append",
        method=insert_on_conflict,
    )
    result = sql.read_sql_table("test_insert_conflict", conn)
    tm.assert_frame_equal(result, expected)
    assert inserted == 2

    # Cleanup
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("test_insert_conflict")


@pytest.mark.db
@pytest.mark.parametrize("conn", postgresql_connectable)
def test_read_view_postgres(conn, request):
    # GH 52969
    conn = request.getfixturevalue(conn)

    from sqlalchemy.engine import Engine
    from sqlalchemy.sql import text

    table_name = f"group_{uuid.uuid4().hex}"
    view_name = f"group_view_{uuid.uuid4().hex}"

    sql_stmt = text(
        f"""
    CREATE TABLE {table_name} (
        group_id INTEGER,
        name TEXT
    );
    INSERT INTO {table_name} VALUES
        (1, 'name');
    CREATE VIEW {view_name}
    AS
    SELECT * FROM {table_name};
    """
    )
    if isinstance(conn, Engine):
        with conn.connect() as con:
            with con.begin():
                con.execute(sql_stmt)
    else:
        with conn.begin():
            conn.execute(sql_stmt)
    result = read_sql_table(view_name, conn)
    expected = DataFrame({"group_id": [1], "name": "name"})
    tm.assert_frame_equal(result, expected)


def test_read_view_sqlite(sqlite_buildin):
    # GH 52969
    create_table = """
CREATE TABLE groups (
   group_id INTEGER,
   name TEXT
);
"""
    insert_into = """
INSERT INTO groups VALUES
    (1, 'name');
"""
    create_view = """
CREATE VIEW group_view
AS
SELECT * FROM groups;
"""
    sqlite_buildin.execute(create_table)
    sqlite_buildin.execute(insert_into)
    sqlite_buildin.execute(create_view)
    result = pd.read_sql("SELECT * FROM group_view", sqlite_buildin)
    expected = DataFrame({"group_id": [1], "name": "name"})
    tm.assert_frame_equal(result, expected)


def test_execute_typeerror(sqlite_iris_engine):
    with pytest.raises(TypeError, match="pandas.io.sql.execute requires a connection"):
        with tm.assert_produces_warning(
            FutureWarning,
            match="`pandas.io.sql.execute` is deprecated and "
            "will be removed in the future version.",
        ):
            sql.execute("select * from iris", sqlite_iris_engine)


def test_execute_deprecated(sqlite_buildin_iris):
    # GH50185
    with tm.assert_produces_warning(
        FutureWarning,
        match="`pandas.io.sql.execute` is deprecated and "
        "will be removed in the future version.",
    ):
        sql.execute("select * from iris", sqlite_buildin_iris)


class MixInBase:
    def teardown_method(self):
        # if setup fails, there may not be a connection to close.
        if hasattr(self, "conn"):
            self.conn.close()
        # use a fresh connection to ensure we can drop all tables.
        try:
            conn = self.connect()
        except (sqlalchemy.exc.OperationalError, sqlite3.OperationalError):
            pass
        else:
            with conn:
                for view in self._get_all_views(conn):
                    self.drop_view(view, conn)
                for tbl in self._get_all_tables(conn):
                    self.drop_table(tbl, conn)


class SQLiteMixIn(MixInBase):
    def connect(self):
        return sqlite3.connect(":memory:")

    def drop_table(self, table_name, conn):
        conn.execute(f"DROP TABLE IF EXISTS {sql._get_valid_sqlite_name(table_name)}")
        conn.commit()

    def _get_all_tables(self, conn):
        c = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [table[0] for table in c.fetchall()]

    def drop_view(self, view_name, conn):
        conn.execute(f"DROP VIEW IF EXISTS {sql._get_valid_sqlite_name(view_name)}")
        conn.commit()

    def _get_all_views(self, conn):
        c = conn.execute("SELECT name FROM sqlite_master WHERE type='view'")
        return [view[0] for view in c.fetchall()]


class SQLAlchemyMixIn(MixInBase):
    @classmethod
    def teardown_class(cls):
        cls.engine.dispose()

    def connect(self):
        return self.engine.connect()

    def drop_table(self, table_name, conn):
        if conn.in_transaction():
            conn.get_transaction().rollback()
        with conn.begin():
            sql.SQLDatabase(conn).drop_table(table_name)

    def _get_all_tables(self, conn):
        from sqlalchemy import inspect

        return inspect(conn).get_table_names()

    def drop_view(self, view_name, conn):
        quoted_view = conn.engine.dialect.identifier_preparer.quote_identifier(
            view_name
        )
        if conn.in_transaction():
            conn.get_transaction().rollback()
        with conn.begin():
            conn.exec_driver_sql(f"DROP VIEW IF EXISTS {quoted_view}")

    def _get_all_views(self, conn):
        from sqlalchemy import inspect

        return inspect(conn).get_view_names()


class PandasSQLTest:
    """
    Base class with common private methods for SQLAlchemy and fallback cases.

    """

    def load_iris_data(self, iris_path):
        self.drop_table("iris", self.conn)
        if isinstance(self.conn, sqlite3.Connection):
            create_and_load_iris_sqlite3(self.conn, iris_path)
        else:
            create_and_load_iris(self.conn, iris_path, self.flavor)

    def load_types_data(self, types_data):
        if self.flavor != "postgresql":
            for entry in types_data:
                entry.pop("DateColWithTz")
        if isinstance(self.conn, sqlite3.Connection):
            types_data = [tuple(entry.values()) for entry in types_data]
            create_and_load_types_sqlite3(self.conn, types_data)
        else:
            create_and_load_types(self.conn, types_data, self.flavor)

    def _read_sql_iris_parameter(self, sql_strings):
        query = sql_strings["read_parameters"][self.flavor]
        params = ("Iris-setosa", 5.1)
        iris_frame = self.pandasSQL.read_query(query, params=params)
        check_iris_frame(iris_frame)

    def _read_sql_iris_named_parameter(self, sql_strings):
        query = sql_strings["read_named_parameters"][self.flavor]
        params = {"name": "Iris-setosa", "length": 5.1}
        iris_frame = self.pandasSQL.read_query(query, params=params)
        check_iris_frame(iris_frame)

    def _read_sql_iris_no_parameter_with_percent(self, sql_strings):
        query = sql_strings["read_no_parameters_with_percent"][self.flavor]
        iris_frame = self.pandasSQL.read_query(query, params=None)
        check_iris_frame(iris_frame)

    def _to_sql_empty(self, test_frame1):
        self.drop_table("test_frame1", self.conn)
        assert self.pandasSQL.to_sql(test_frame1.iloc[:0], "test_frame1") == 0

    def _to_sql_with_sql_engine(self, test_frame1, engine="auto", **engine_kwargs):
        """`to_sql` with the `engine` param"""
        # mostly copied from this class's `_to_sql()` method
        self.drop_table("test_frame1", self.conn)

        assert (
            self.pandasSQL.to_sql(
                test_frame1, "test_frame1", engine=engine, **engine_kwargs
            )
            == 4
        )
        assert self.pandasSQL.has_table("test_frame1")

        num_entries = len(test_frame1)
        num_rows = count_rows(self.conn, "test_frame1")
        assert num_rows == num_entries

        # Nuke table
        self.drop_table("test_frame1", self.conn)

    def _roundtrip(self, test_frame1):
        self.drop_table("test_frame_roundtrip", self.conn)
        assert self.pandasSQL.to_sql(test_frame1, "test_frame_roundtrip") == 4
        result = self.pandasSQL.read_query("SELECT * FROM test_frame_roundtrip")

        result.set_index("level_0", inplace=True)
        # result.index.astype(int)

        result.index.name = None

        tm.assert_frame_equal(result, test_frame1)

    def _execute_sql(self):
        # drop_sql = "DROP TABLE IF EXISTS test"  # should already be done
        iris_results = self.pandasSQL.execute("SELECT * FROM iris")
        row = iris_results.fetchone()
        tm.equalContents(row, [5.1, 3.5, 1.4, 0.2, "Iris-setosa"])

    def _to_sql_save_index(self):
        df = DataFrame.from_records(
            [(1, 2.1, "line1"), (2, 1.5, "line2")], columns=["A", "B", "C"], index=["A"]
        )
        assert self.pandasSQL.to_sql(df, "test_to_sql_saves_index") == 2
        ix_cols = self._get_index_columns("test_to_sql_saves_index")
        assert ix_cols == [["A"]]

    def _transaction_test(self):
        with self.pandasSQL.run_transaction() as trans:
            stmt = "CREATE TABLE test_trans (A INT, B TEXT)"
            if isinstance(self.pandasSQL, SQLiteDatabase):
                trans.execute(stmt)
            else:
                from sqlalchemy import text

                stmt = text(stmt)
                trans.execute(stmt)

        class DummyException(Exception):
            pass

        # Make sure when transaction is rolled back, no rows get inserted
        ins_sql = "INSERT INTO test_trans (A,B) VALUES (1, 'blah')"
        if isinstance(self.pandasSQL, SQLDatabase):
            from sqlalchemy import text

            ins_sql = text(ins_sql)
        try:
            with self.pandasSQL.run_transaction() as trans:
                trans.execute(ins_sql)
                raise DummyException("error")
        except DummyException:
            # ignore raised exception
            pass
        res = self.pandasSQL.read_query("SELECT * FROM test_trans")
        assert len(res) == 0

        # Make sure when transaction is committed, rows do get inserted
        with self.pandasSQL.run_transaction() as trans:
            trans.execute(ins_sql)
        res2 = self.pandasSQL.read_query("SELECT * FROM test_trans")
        assert len(res2) == 1


# -----------------------------------------------------------------------------
# -- Testing the public API


class _TestSQLApi(PandasSQLTest):
    """
    Base class to test the public API.

    From this two classes are derived to run these tests for both the
    sqlalchemy mode (`TestSQLApi`) and the fallback mode
    (`TestSQLiteFallbackApi`).  These tests are run with sqlite3. Specific
    tests for the different sql flavours are included in `_TestSQLAlchemy`.

    Notes:
    flavor can always be passed even in SQLAlchemy mode,
    should be correctly ignored.

    we don't use drop_table because that isn't part of the public api

    """

    flavor = "sqlite"
    mode: str

    @pytest.fixture(autouse=True)
    def setup_method(self, iris_path, types_data):
        self.conn = self.connect()
        self.load_iris_data(iris_path)
        self.load_types_data(types_data)
        self.load_test_data_and_sql()

    def load_test_data_and_sql(self):
        create_and_load_iris_view(self.conn)

    def test_read_sql_view(self):
        iris_frame = sql.read_sql_query("SELECT * FROM iris_view", self.conn)
        check_iris_frame(iris_frame)

    def test_read_sql_with_chunksize_no_result(self):
        query = "SELECT * FROM iris_view WHERE SepalLength < 0.0"
        with_batch = sql.read_sql_query(query, self.conn, chunksize=5)
        without_batch = sql.read_sql_query(query, self.conn)
        tm.assert_frame_equal(concat(with_batch), without_batch)

    def test_to_sql(self, test_frame1):
        sql.to_sql(test_frame1, "test_frame1", self.conn)
        assert sql.has_table("test_frame1", self.conn)

    def test_to_sql_fail(self, test_frame1):
        sql.to_sql(test_frame1, "test_frame2", self.conn, if_exists="fail")
        assert sql.has_table("test_frame2", self.conn)

        msg = "Table 'test_frame2' already exists"
        with pytest.raises(ValueError, match=msg):
            sql.to_sql(test_frame1, "test_frame2", self.conn, if_exists="fail")

    def test_to_sql_replace(self, test_frame1):
        sql.to_sql(test_frame1, "test_frame3", self.conn, if_exists="fail")
        # Add to table again
        sql.to_sql(test_frame1, "test_frame3", self.conn, if_exists="replace")
        assert sql.has_table("test_frame3", self.conn)

        num_entries = len(test_frame1)
        num_rows = count_rows(self.conn, "test_frame3")

        assert num_rows == num_entries

    def test_to_sql_append(self, test_frame1):
        assert sql.to_sql(test_frame1, "test_frame4", self.conn, if_exists="fail") == 4

        # Add to table again
        assert (
            sql.to_sql(test_frame1, "test_frame4", self.conn, if_exists="append") == 4
        )
        assert sql.has_table("test_frame4", self.conn)

        num_entries = 2 * len(test_frame1)
        num_rows = count_rows(self.conn, "test_frame4")

        assert num_rows == num_entries

    def test_to_sql_type_mapping(self, test_frame3):
        sql.to_sql(test_frame3, "test_frame5", self.conn, index=False)
        result = sql.read_sql("SELECT * FROM test_frame5", self.conn)

        tm.assert_frame_equal(test_frame3, result)

    def test_to_sql_series(self):
        s = Series(np.arange(5, dtype="int64"), name="series")
        sql.to_sql(s, "test_series", self.conn, index=False)
        s2 = sql.read_sql_query("SELECT * FROM test_series", self.conn)
        tm.assert_frame_equal(s.to_frame(), s2)

    def test_roundtrip(self, test_frame1):
        sql.to_sql(test_frame1, "test_frame_roundtrip", con=self.conn)
        result = sql.read_sql_query("SELECT * FROM test_frame_roundtrip", con=self.conn)

        # HACK!
        result.index = test_frame1.index
        result.set_index("level_0", inplace=True)
        result.index.astype(int)
        result.index.name = None
        tm.assert_frame_equal(result, test_frame1)

    def test_roundtrip_chunksize(self, test_frame1):
        sql.to_sql(
            test_frame1,
            "test_frame_roundtrip",
            con=self.conn,
            index=False,
            chunksize=2,
        )
        result = sql.read_sql_query("SELECT * FROM test_frame_roundtrip", con=self.conn)
        tm.assert_frame_equal(result, test_frame1)

    def test_execute_sql(self):
        # drop_sql = "DROP TABLE IF EXISTS test"  # should already be done
        with sql.pandasSQL_builder(self.conn) as pandas_sql:
            iris_results = pandas_sql.execute("SELECT * FROM iris")
        row = iris_results.fetchone()
        tm.equalContents(row, [5.1, 3.5, 1.4, 0.2, "Iris-setosa"])

    def test_date_parsing(self):
        # Test date parsing in read_sql
        # No Parsing
        df = sql.read_sql_query("SELECT * FROM types", self.conn)
        assert not issubclass(df.DateCol.dtype.type, np.datetime64)

        df = sql.read_sql_query(
            "SELECT * FROM types", self.conn, parse_dates=["DateCol"]
        )
        assert issubclass(df.DateCol.dtype.type, np.datetime64)
        assert df.DateCol.tolist() == [
            Timestamp(2000, 1, 3, 0, 0, 0),
            Timestamp(2000, 1, 4, 0, 0, 0),
        ]

        df = sql.read_sql_query(
            "SELECT * FROM types",
            self.conn,
            parse_dates={"DateCol": "%Y-%m-%d %H:%M:%S"},
        )
        assert issubclass(df.DateCol.dtype.type, np.datetime64)
        assert df.DateCol.tolist() == [
            Timestamp(2000, 1, 3, 0, 0, 0),
            Timestamp(2000, 1, 4, 0, 0, 0),
        ]

        df = sql.read_sql_query(
            "SELECT * FROM types", self.conn, parse_dates=["IntDateCol"]
        )
        assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
        assert df.IntDateCol.tolist() == [
            Timestamp(1986, 12, 25, 0, 0, 0),
            Timestamp(2013, 1, 1, 0, 0, 0),
        ]

        df = sql.read_sql_query(
            "SELECT * FROM types", self.conn, parse_dates={"IntDateCol": "s"}
        )
        assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
        assert df.IntDateCol.tolist() == [
            Timestamp(1986, 12, 25, 0, 0, 0),
            Timestamp(2013, 1, 1, 0, 0, 0),
        ]

        df = sql.read_sql_query(
            "SELECT * FROM types",
            self.conn,
            parse_dates={"IntDateOnlyCol": "%Y%m%d"},
        )
        assert issubclass(df.IntDateOnlyCol.dtype.type, np.datetime64)
        assert df.IntDateOnlyCol.tolist() == [
            Timestamp("2010-10-10"),
            Timestamp("2010-12-12"),
        ]

    @pytest.mark.parametrize("error", ["ignore", "raise", "coerce"])
    @pytest.mark.parametrize(
        "read_sql, text, mode",
        [
            (sql.read_sql, "SELECT * FROM types", ("sqlalchemy", "fallback")),
            (sql.read_sql, "types", ("sqlalchemy")),
            (
                sql.read_sql_query,
                "SELECT * FROM types",
                ("sqlalchemy", "fallback"),
            ),
            (sql.read_sql_table, "types", ("sqlalchemy")),
        ],
    )
    def test_custom_dateparsing_error(
        self, read_sql, text, mode, error, types_data_frame
    ):
        if self.mode in mode:
            expected = types_data_frame.astype({"DateCol": "datetime64[ns]"})

            result = read_sql(
                text,
                con=self.conn,
                parse_dates={
                    "DateCol": {"errors": error},
                },
            )

            tm.assert_frame_equal(result, expected)

    def test_date_and_index(self):
        # Test case where same column appears in parse_date and index_col

        df = sql.read_sql_query(
            "SELECT * FROM types",
            self.conn,
            index_col="DateCol",
            parse_dates=["DateCol", "IntDateCol"],
        )

        assert issubclass(df.index.dtype.type, np.datetime64)
        assert issubclass(df.IntDateCol.dtype.type, np.datetime64)

    def test_timedelta(self):
        # see #6921
        df = to_timedelta(Series(["00:00:01", "00:00:03"], name="foo")).to_frame()
        with tm.assert_produces_warning(UserWarning):
            result_count = df.to_sql(name="test_timedelta", con=self.conn)
        assert result_count == 2
        result = sql.read_sql_query("SELECT * FROM test_timedelta", self.conn)
        tm.assert_series_equal(result["foo"], df["foo"].view("int64"))

    def test_complex_raises(self):
        df = DataFrame({"a": [1 + 1j, 2j]})
        msg = "Complex datatypes not supported"
        with pytest.raises(ValueError, match=msg):
            assert df.to_sql("test_complex", con=self.conn) is None

    @pytest.mark.parametrize(
        "index_name,index_label,expected",
        [
            # no index name, defaults to 'index'
            (None, None, "index"),
            # specifying index_label
            (None, "other_label", "other_label"),
            # using the index name
            ("index_name", None, "index_name"),
            # has index name, but specifying index_label
            ("index_name", "other_label", "other_label"),
            # index name is integer
            (0, None, "0"),
            # index name is None but index label is integer
            (None, 0, "0"),
        ],
    )
    def test_to_sql_index_label(self, index_name, index_label, expected):
        temp_frame = DataFrame({"col1": range(4)})
        temp_frame.index.name = index_name
        query = "SELECT * FROM test_index_label"
        sql.to_sql(temp_frame, "test_index_label", self.conn, index_label=index_label)
        frame = sql.read_sql_query(query, self.conn)
        assert frame.columns[0] == expected

    def test_to_sql_index_label_multiindex(self):
        expected_row_count = 4
        temp_frame = DataFrame(
            {"col1": range(4)},
            index=MultiIndex.from_product([("A0", "A1"), ("B0", "B1")]),
        )

        # no index name, defaults to 'level_0' and 'level_1'
        result = sql.to_sql(temp_frame, "test_index_label", self.conn)
        assert result == expected_row_count
        frame = sql.read_sql_query("SELECT * FROM test_index_label", self.conn)
        assert frame.columns[0] == "level_0"
        assert frame.columns[1] == "level_1"

        # specifying index_label
        result = sql.to_sql(
            temp_frame,
            "test_index_label",
            self.conn,
            if_exists="replace",
            index_label=["A", "B"],
        )
        assert result == expected_row_count
        frame = sql.read_sql_query("SELECT * FROM test_index_label", self.conn)
        assert frame.columns[:2].tolist() == ["A", "B"]

        # using the index name
        temp_frame.index.names = ["A", "B"]
        result = sql.to_sql(
            temp_frame, "test_index_label", self.conn, if_exists="replace"
        )
        assert result == expected_row_count
        frame = sql.read_sql_query("SELECT * FROM test_index_label", self.conn)
        assert frame.columns[:2].tolist() == ["A", "B"]

        # has index name, but specifying index_label
        result = sql.to_sql(
            temp_frame,
            "test_index_label",
            self.conn,
            if_exists="replace",
            index_label=["C", "D"],
        )
        assert result == expected_row_count
        frame = sql.read_sql_query("SELECT * FROM test_index_label", self.conn)
        assert frame.columns[:2].tolist() == ["C", "D"]

        msg = "Length of 'index_label' should match number of levels, which is 2"
        with pytest.raises(ValueError, match=msg):
            sql.to_sql(
                temp_frame,
                "test_index_label",
                self.conn,
                if_exists="replace",
                index_label="C",
            )

    def test_multiindex_roundtrip(self):
        df = DataFrame.from_records(
            [(1, 2.1, "line1"), (2, 1.5, "line2")],
            columns=["A", "B", "C"],
            index=["A", "B"],
        )

        df.to_sql(name="test_multiindex_roundtrip", con=self.conn)
        result = sql.read_sql_query(
            "SELECT * FROM test_multiindex_roundtrip", self.conn, index_col=["A", "B"]
        )
        tm.assert_frame_equal(df, result, check_index_type=True)

    @pytest.mark.parametrize(
        "dtype",
        [
            None,
            int,
            float,
            {"A": int, "B": float},
        ],
    )
    def test_dtype_argument(self, dtype):
        # GH10285 Add dtype argument to read_sql_query
        df = DataFrame([[1.2, 3.4], [5.6, 7.8]], columns=["A", "B"])
        assert df.to_sql(name="test_dtype_argument", con=self.conn) == 2

        expected = df.astype(dtype)
        result = sql.read_sql_query(
            "SELECT A, B FROM test_dtype_argument", con=self.conn, dtype=dtype
        )

        tm.assert_frame_equal(result, expected)

    def test_integer_col_names(self):
        df = DataFrame([[1, 2], [3, 4]], columns=[0, 1])
        sql.to_sql(df, "test_frame_integer_col_names", self.conn, if_exists="replace")

    def test_get_schema(self, test_frame1):
        create_sql = sql.get_schema(test_frame1, "test", con=self.conn)
        assert "CREATE" in create_sql

    def test_get_schema_with_schema(self, test_frame1):
        # GH28486
        create_sql = sql.get_schema(test_frame1, "test", con=self.conn, schema="pypi")
        assert "CREATE TABLE pypi." in create_sql

    def test_get_schema_dtypes(self):
        if self.mode == "sqlalchemy":
            from sqlalchemy import Integer

            dtype = Integer
        else:
            dtype = "INTEGER"

        float_frame = DataFrame({"a": [1.1, 1.2], "b": [2.1, 2.2]})
        create_sql = sql.get_schema(
            float_frame, "test", con=self.conn, dtype={"b": dtype}
        )
        assert "CREATE" in create_sql
        assert "INTEGER" in create_sql

    def test_get_schema_keys(self, test_frame1):
        frame = DataFrame({"Col1": [1.1, 1.2], "Col2": [2.1, 2.2]})
        create_sql = sql.get_schema(frame, "test", con=self.conn, keys="Col1")
        constraint_sentence = 'CONSTRAINT test_pk PRIMARY KEY ("Col1")'
        assert constraint_sentence in create_sql

        # multiple columns as key (GH10385)
        create_sql = sql.get_schema(test_frame1, "test", con=self.conn, keys=["A", "B"])
        constraint_sentence = 'CONSTRAINT test_pk PRIMARY KEY ("A", "B")'
        assert constraint_sentence in create_sql

    def test_chunksize_read(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((22, 5)), columns=list("abcde")
        )
        df.to_sql(name="test_chunksize", con=self.conn, index=False)

        # reading the query in one time
        res1 = sql.read_sql_query("select * from test_chunksize", self.conn)

        # reading the query in chunks with read_sql_query
        res2 = DataFrame()
        i = 0
        sizes = [5, 5, 5, 5, 2]

        for chunk in sql.read_sql_query(
            "select * from test_chunksize", self.conn, chunksize=5
        ):
            res2 = concat([res2, chunk], ignore_index=True)
            assert len(chunk) == sizes[i]
            i += 1

        tm.assert_frame_equal(res1, res2)

        # reading the query in chunks with read_sql_query
        if self.mode == "sqlalchemy":
            res3 = DataFrame()
            i = 0
            sizes = [5, 5, 5, 5, 2]

            for chunk in sql.read_sql_table("test_chunksize", self.conn, chunksize=5):
                res3 = concat([res3, chunk], ignore_index=True)
                assert len(chunk) == sizes[i]
                i += 1

            tm.assert_frame_equal(res1, res3)

    def test_categorical(self):
        # GH8624
        # test that categorical gets written correctly as dense column
        df = DataFrame(
            {
                "person_id": [1, 2, 3],
                "person_name": ["John P. Doe", "Jane Dove", "John P. Doe"],
            }
        )
        df2 = df.copy()
        df2["person_name"] = df2["person_name"].astype("category")

        df2.to_sql(name="test_categorical", con=self.conn, index=False)
        res = sql.read_sql_query("SELECT * FROM test_categorical", self.conn)

        tm.assert_frame_equal(res, df)

    def test_unicode_column_name(self):
        # GH 11431
        df = DataFrame([[1, 2], [3, 4]], columns=["\xe9", "b"])
        df.to_sql(name="test_unicode", con=self.conn, index=False)

    def test_escaped_table_name(self):
        # GH 13206
        df = DataFrame({"A": [0, 1, 2], "B": [0.2, np.nan, 5.6]})
        df.to_sql(name="d1187b08-4943-4c8d-a7f6", con=self.conn, index=False)

        res = sql.read_sql_query("SELECT * FROM `d1187b08-4943-4c8d-a7f6`", self.conn)

        tm.assert_frame_equal(res, df)

    def test_read_sql_duplicate_columns(self):
        # GH#53117
        df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3], "c": 1})
        df.to_sql(name="test_table", con=self.conn, index=False)

        result = pd.read_sql("SELECT a, b, a +1 as a, c FROM test_table;", self.conn)
        expected = DataFrame(
            [[1, 0.1, 2, 1], [2, 0.2, 3, 1], [3, 0.3, 4, 1]],
            columns=["a", "b", "a", "c"],
        )
        tm.assert_frame_equal(result, expected)


@pytest.mark.skipif(not SQLALCHEMY_INSTALLED, reason="SQLAlchemy not installed")
class TestSQLApi(SQLAlchemyMixIn, _TestSQLApi):
    """
    Test the public API as it would be used directly

    Tests for `read_sql_table` are included here, as this is specific for the
    sqlalchemy mode.

    """

    flavor = "sqlite"
    mode = "sqlalchemy"

    @classmethod
    def setup_class(cls):
        cls.engine = sqlalchemy.create_engine("sqlite:///:memory:")

    def test_read_table_columns(self, test_frame1):
        # test columns argument in read_table
        sql.to_sql(test_frame1, "test_frame", self.conn)

        cols = ["A", "B"]
        result = sql.read_sql_table("test_frame", self.conn, columns=cols)
        assert result.columns.tolist() == cols

    def test_read_table_index_col(self, test_frame1):
        # test columns argument in read_table
        sql.to_sql(test_frame1, "test_frame", self.conn)

        result = sql.read_sql_table("test_frame", self.conn, index_col="index")
        assert result.index.names == ["index"]

        result = sql.read_sql_table("test_frame", self.conn, index_col=["A", "B"])
        assert result.index.names == ["A", "B"]

        result = sql.read_sql_table(
            "test_frame", self.conn, index_col=["A", "B"], columns=["C", "D"]
        )
        assert result.index.names == ["A", "B"]
        assert result.columns.tolist() == ["C", "D"]

    def test_read_sql_delegate(self):
        iris_frame1 = sql.read_sql_query("SELECT * FROM iris", self.conn)
        iris_frame2 = sql.read_sql("SELECT * FROM iris", self.conn)
        tm.assert_frame_equal(iris_frame1, iris_frame2)

        iris_frame1 = sql.read_sql_table("iris", self.conn)
        iris_frame2 = sql.read_sql("iris", self.conn)
        tm.assert_frame_equal(iris_frame1, iris_frame2)

    def test_not_reflect_all_tables(self):
        from sqlalchemy import text
        from sqlalchemy.engine import Engine

        # create invalid table
        query_list = [
            text("CREATE TABLE invalid (x INTEGER, y UNKNOWN);"),
            text("CREATE TABLE other_table (x INTEGER, y INTEGER);"),
        ]
        for query in query_list:
            if isinstance(self.conn, Engine):
                with self.conn.connect() as conn:
                    with conn.begin():
                        conn.execute(query)
            else:
                with self.conn.begin():
                    self.conn.execute(query)

        with tm.assert_produces_warning(None):
            sql.read_sql_table("other_table", self.conn)
            sql.read_sql_query("SELECT * FROM other_table", self.conn)

    def test_warning_case_insensitive_table_name(self, test_frame1):
        # see gh-7815
        with tm.assert_produces_warning(
            UserWarning,
            match=(
                r"The provided table name 'TABLE1' is not found exactly as such in "
                r"the database after writing the table, possibly due to case "
                r"sensitivity issues. Consider using lower case table names."
            ),
        ):
            sql.SQLDatabase(self.conn).check_case_sensitive("TABLE1", "")

        # Test that the warning is certainly NOT triggered in a normal case.
        with tm.assert_produces_warning(None):
            test_frame1.to_sql(name="CaseSensitive", con=self.conn)

    def _get_index_columns(self, tbl_name):
        from sqlalchemy.engine import reflection

        insp = reflection.Inspector.from_engine(self.conn)
        ixs = insp.get_indexes("test_index_saved")
        ixs = [i["column_names"] for i in ixs]
        return ixs

    def test_sqlalchemy_type_mapping(self):
        from sqlalchemy import TIMESTAMP

        # Test Timestamp objects (no datetime64 because of timezone) (GH9085)
        df = DataFrame(
            {"time": to_datetime(["2014-12-12 01:54", "2014-12-11 02:54"], utc=True)}
        )
        db = sql.SQLDatabase(self.conn)
        table = sql.SQLTable("test_type", db, frame=df)
        # GH 9086: TIMESTAMP is the suggested type for datetimes with timezones
        assert isinstance(table.table.c["time"].type, TIMESTAMP)

    @pytest.mark.parametrize(
        "integer, expected",
        [
            ("int8", "SMALLINT"),
            ("Int8", "SMALLINT"),
            ("uint8", "SMALLINT"),
            ("UInt8", "SMALLINT"),
            ("int16", "SMALLINT"),
            ("Int16", "SMALLINT"),
            ("uint16", "INTEGER"),
            ("UInt16", "INTEGER"),
            ("int32", "INTEGER"),
            ("Int32", "INTEGER"),
            ("uint32", "BIGINT"),
            ("UInt32", "BIGINT"),
            ("int64", "BIGINT"),
            ("Int64", "BIGINT"),
            (int, "BIGINT" if np.dtype(int).name == "int64" else "INTEGER"),
        ],
    )
    def test_sqlalchemy_integer_mapping(self, integer, expected):
        # GH35076 Map pandas integer to optimal SQLAlchemy integer type
        df = DataFrame([0, 1], columns=["a"], dtype=integer)
        db = sql.SQLDatabase(self.conn)
        table = sql.SQLTable("test_type", db, frame=df)

        result = str(table.table.c.a.type)
        assert result == expected

    @pytest.mark.parametrize("integer", ["uint64", "UInt64"])
    def test_sqlalchemy_integer_overload_mapping(self, integer):
        # GH35076 Map pandas integer to optimal SQLAlchemy integer type
        df = DataFrame([0, 1], columns=["a"], dtype=integer)
        db = sql.SQLDatabase(self.conn)
        with pytest.raises(
            ValueError, match="Unsigned 64 bit integer datatype is not supported"
        ):
            sql.SQLTable("test_type", db, frame=df)

    def test_database_uri_string(self, test_frame1):
        # Test read_sql and .to_sql method with a database URI (GH10654)
        # db_uri = 'sqlite:///:memory:' # raises
        # sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) near
        # "iris": syntax error [SQL: 'iris']
        with tm.ensure_clean() as name:
            db_uri = "sqlite:///" + name
            table = "iris"
            test_frame1.to_sql(name=table, con=db_uri, if_exists="replace", index=False)
            test_frame2 = sql.read_sql(table, db_uri)
            test_frame3 = sql.read_sql_table(table, db_uri)
            query = "SELECT * FROM iris"
            test_frame4 = sql.read_sql_query(query, db_uri)
        tm.assert_frame_equal(test_frame1, test_frame2)
        tm.assert_frame_equal(test_frame1, test_frame3)
        tm.assert_frame_equal(test_frame1, test_frame4)

    @td.skip_if_installed("pg8000")
    def test_pg8000_sqlalchemy_passthrough_error(self):
        # using driver that will not be installed on CI to trigger error
        # in sqlalchemy.create_engine -> test passing of this error to user
        db_uri = "postgresql+pg8000://user:pass@host/dbname"
        with pytest.raises(ImportError, match="pg8000"):
            sql.read_sql("select * from table", db_uri)

    def test_query_by_text_obj(self):
        # WIP : GH10846
        from sqlalchemy import text

        name_text = text("select * from iris where name=:name")
        iris_df = sql.read_sql(name_text, self.conn, params={"name": "Iris-versicolor"})
        all_names = set(iris_df["Name"])
        assert all_names == {"Iris-versicolor"}

    def test_query_by_select_obj(self):
        # WIP : GH10846
        from sqlalchemy import (
            bindparam,
            select,
        )

        iris = iris_table_metadata(self.flavor)
        name_select = select(iris).where(iris.c.Name == bindparam("name"))
        iris_df = sql.read_sql(name_select, self.conn, params={"name": "Iris-setosa"})
        all_names = set(iris_df["Name"])
        assert all_names == {"Iris-setosa"}

    def test_column_with_percentage(self):
        # GH 37157
        df = DataFrame({"A": [0, 1, 2], "%_variation": [3, 4, 5]})
        df.to_sql(name="test_column_percentage", con=self.conn, index=False)

        res = sql.read_sql_table("test_column_percentage", self.conn)

        tm.assert_frame_equal(res, df)


class TestSQLiteFallbackApi(SQLiteMixIn, _TestSQLApi):
    """
    Test the public sqlite connection fallback API

    """

    flavor = "sqlite"
    mode = "fallback"

    def connect(self, database=":memory:"):
        return sqlite3.connect(database)

    def test_sql_open_close(self, test_frame3):
        # Test if the IO in the database still work if the connection closed
        # between the writing and reading (as in many real situations).

        with tm.ensure_clean() as name:
            with closing(self.connect(name)) as conn:
                assert (
                    sql.to_sql(test_frame3, "test_frame3_legacy", conn, index=False)
                    == 4
                )

            with closing(self.connect(name)) as conn:
                result = sql.read_sql_query("SELECT * FROM test_frame3_legacy;", conn)

        tm.assert_frame_equal(test_frame3, result)

    @pytest.mark.skipif(SQLALCHEMY_INSTALLED, reason="SQLAlchemy is installed")
    def test_con_string_import_error(self):
        conn = "mysql://root@localhost/pandas"
        msg = "Using URI string without sqlalchemy installed"
        with pytest.raises(ImportError, match=msg):
            sql.read_sql("SELECT * FROM iris", conn)

    @pytest.mark.skipif(SQLALCHEMY_INSTALLED, reason="SQLAlchemy is installed")
    def test_con_unknown_dbapi2_class_does_not_error_without_sql_alchemy_installed(
        self,
    ):
        class MockSqliteConnection:
            def __init__(self, *args, **kwargs) -> None:
                self.conn = sqlite3.Connection(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self.conn, name)

            def close(self):
                self.conn.close()

        with contextlib.closing(MockSqliteConnection(":memory:")) as conn:
            with tm.assert_produces_warning(UserWarning):
                sql.read_sql("SELECT 1", conn)

    def test_read_sql_delegate(self):
        iris_frame1 = sql.read_sql_query("SELECT * FROM iris", self.conn)
        iris_frame2 = sql.read_sql("SELECT * FROM iris", self.conn)
        tm.assert_frame_equal(iris_frame1, iris_frame2)

        msg = "Execution failed on sql 'iris': near \"iris\": syntax error"
        with pytest.raises(sql.DatabaseError, match=msg):
            sql.read_sql("iris", self.conn)

    def test_get_schema2(self, test_frame1):
        # without providing a connection object (available for backwards comp)
        create_sql = sql.get_schema(test_frame1, "test")
        assert "CREATE" in create_sql

    def _get_sqlite_column_type(self, schema, column):
        for col in schema.split("\n"):
            if col.split()[0].strip('"') == column:
                return col.split()[1]
        raise ValueError(f"Column {column} not found")

    def test_sqlite_type_mapping(self):
        # Test Timestamp objects (no datetime64 because of timezone) (GH9085)
        df = DataFrame(
            {"time": to_datetime(["2014-12-12 01:54", "2014-12-11 02:54"], utc=True)}
        )
        db = sql.SQLiteDatabase(self.conn)
        table = sql.SQLiteTable("test_type", db, frame=df)
        schema = table.sql_schema()
        assert self._get_sqlite_column_type(schema, "time") == "TIMESTAMP"


# -----------------------------------------------------------------------------
# -- Database flavor specific tests


@pytest.mark.skipif(not SQLALCHEMY_INSTALLED, reason="SQLAlchemy not installed")
class _TestSQLAlchemy(SQLAlchemyMixIn, PandasSQLTest):
    """
    Base class for testing the sqlalchemy backend.

    Subclasses for specific database types are created below. Tests that
    deviate for each flavor are overwritten there.

    """

    flavor: str

    @classmethod
    def setup_class(cls):
        cls.setup_driver()
        cls.setup_engine()

    @pytest.fixture(autouse=True)
    def setup_method(self, iris_path, types_data):
        try:
            self.conn = self.engine.connect()
            self.pandasSQL = sql.SQLDatabase(self.conn)
        except sqlalchemy.exc.OperationalError:
            pytest.skip(f"Can't connect to {self.flavor} server")
        self.load_iris_data(iris_path)
        self.load_types_data(types_data)

    @classmethod
    def setup_driver(cls):
        raise NotImplementedError()

    @classmethod
    def setup_engine(cls):
        raise NotImplementedError()

    def test_read_sql_parameter(self, sql_strings):
        self._read_sql_iris_parameter(sql_strings)

    def test_read_sql_named_parameter(self, sql_strings):
        self._read_sql_iris_named_parameter(sql_strings)

    def test_to_sql_empty(self, test_frame1):
        self._to_sql_empty(test_frame1)

    def test_create_table(self):
        from sqlalchemy import inspect

        temp_conn = self.connect()
        temp_frame = DataFrame(
            {"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]}
        )
        with sql.SQLDatabase(temp_conn, need_transaction=True) as pandasSQL:
            assert pandasSQL.to_sql(temp_frame, "temp_frame") == 4

        insp = inspect(temp_conn)
        assert insp.has_table("temp_frame")

        # Cleanup
        with sql.SQLDatabase(temp_conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("temp_frame")

    def test_drop_table(self):
        from sqlalchemy import inspect

        temp_conn = self.connect()
        temp_frame = DataFrame(
            {"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]}
        )
        pandasSQL = sql.SQLDatabase(temp_conn)
        assert pandasSQL.to_sql(temp_frame, "temp_frame") == 4

        insp = inspect(temp_conn)
        assert insp.has_table("temp_frame")

        pandasSQL.drop_table("temp_frame")
        try:
            insp.clear_cache()  # needed with SQLAlchemy 2.0, unavailable prior
        except AttributeError:
            pass
        assert not insp.has_table("temp_frame")

    def test_roundtrip(self, test_frame1):
        self._roundtrip(test_frame1)

    def test_execute_sql(self):
        self._execute_sql()

    def test_read_table(self):
        iris_frame = sql.read_sql_table("iris", con=self.conn)
        check_iris_frame(iris_frame)

    def test_read_table_columns(self):
        iris_frame = sql.read_sql_table(
            "iris", con=self.conn, columns=["SepalLength", "SepalLength"]
        )
        tm.equalContents(iris_frame.columns.values, ["SepalLength", "SepalLength"])

    def test_read_table_absent_raises(self):
        msg = "Table this_doesnt_exist not found"
        with pytest.raises(ValueError, match=msg):
            sql.read_sql_table("this_doesnt_exist", con=self.conn)

    def test_default_type_conversion(self):
        df = sql.read_sql_table("types", self.conn)

        assert issubclass(df.FloatCol.dtype.type, np.floating)
        assert issubclass(df.IntCol.dtype.type, np.integer)
        assert issubclass(df.BoolCol.dtype.type, np.bool_)

        # Int column with NA values stays as float
        assert issubclass(df.IntColWithNull.dtype.type, np.floating)
        # Bool column with NA values becomes object
        assert issubclass(df.BoolColWithNull.dtype.type, object)

    def test_bigint(self):
        # int64 should be converted to BigInteger, GH7433
        df = DataFrame(data={"i64": [2**62]})
        assert df.to_sql(name="test_bigint", con=self.conn, index=False) == 1
        result = sql.read_sql_table("test_bigint", self.conn)

        tm.assert_frame_equal(df, result)

    def test_default_date_load(self):
        df = sql.read_sql_table("types", self.conn)

        # IMPORTANT - sqlite has no native date type, so shouldn't parse, but
        # MySQL SHOULD be converted.
        assert issubclass(df.DateCol.dtype.type, np.datetime64)

    def test_datetime_with_timezone(self, request):
        # edge case that converts postgresql datetime with time zone types
        # to datetime64[ns,psycopg2.tz.FixedOffsetTimezone..], which is ok
        # but should be more natural, so coerce to datetime64[ns] for now

        def check(col):
            # check that a column is either datetime64[ns]
            # or datetime64[ns, UTC]
            if lib.is_np_dtype(col.dtype, "M"):
                # "2000-01-01 00:00:00-08:00" should convert to
                # "2000-01-01 08:00:00"
                assert col[0] == Timestamp("2000-01-01 08:00:00")

                # "2000-06-01 00:00:00-07:00" should convert to
                # "2000-06-01 07:00:00"
                assert col[1] == Timestamp("2000-06-01 07:00:00")

            elif isinstance(col.dtype, DatetimeTZDtype):
                assert str(col.dt.tz) == "UTC"

                # "2000-01-01 00:00:00-08:00" should convert to
                # "2000-01-01 08:00:00"
                # "2000-06-01 00:00:00-07:00" should convert to
                # "2000-06-01 07:00:00"
                # GH 6415
                expected_data = [
                    Timestamp("2000-01-01 08:00:00", tz="UTC"),
                    Timestamp("2000-06-01 07:00:00", tz="UTC"),
                ]
                expected = Series(expected_data, name=col.name)
                tm.assert_series_equal(col, expected)

            else:
                raise AssertionError(
                    f"DateCol loaded with incorrect type -> {col.dtype}"
                )

        # GH11216
        df = read_sql_query("select * from types", self.conn)
        if not hasattr(df, "DateColWithTz"):
            request.node.add_marker(
                pytest.mark.xfail(reason="no column with datetime with time zone")
            )

        # this is parsed on Travis (linux), but not on macosx for some reason
        # even with the same versions of psycopg2 & sqlalchemy, possibly a
        # Postgresql server version difference
        col = df.DateColWithTz
        assert isinstance(col.dtype, DatetimeTZDtype)

        df = read_sql_query(
            "select * from types", self.conn, parse_dates=["DateColWithTz"]
        )
        if not hasattr(df, "DateColWithTz"):
            request.node.add_marker(
                pytest.mark.xfail(reason="no column with datetime with time zone")
            )
        col = df.DateColWithTz
        assert isinstance(col.dtype, DatetimeTZDtype)
        assert str(col.dt.tz) == "UTC"
        check(df.DateColWithTz)

        df = concat(
            list(read_sql_query("select * from types", self.conn, chunksize=1)),
            ignore_index=True,
        )
        col = df.DateColWithTz
        assert isinstance(col.dtype, DatetimeTZDtype)
        assert str(col.dt.tz) == "UTC"
        expected = sql.read_sql_table("types", self.conn)
        col = expected.DateColWithTz
        assert isinstance(col.dtype, DatetimeTZDtype)
        tm.assert_series_equal(df.DateColWithTz, expected.DateColWithTz)

        # xref #7139
        # this might or might not be converted depending on the postgres driver
        df = sql.read_sql_table("types", self.conn)
        check(df.DateColWithTz)

    def test_datetime_with_timezone_roundtrip(self):
        # GH 9086
        # Write datetimetz data to a db and read it back
        # For dbs that support timestamps with timezones, should get back UTC
        # otherwise naive data should be returned
        expected = DataFrame(
            {"A": date_range("2013-01-01 09:00:00", periods=3, tz="US/Pacific")}
        )
        assert expected.to_sql(name="test_datetime_tz", con=self.conn, index=False) == 3

        if self.flavor == "postgresql":
            # SQLAlchemy "timezones" (i.e. offsets) are coerced to UTC
            expected["A"] = expected["A"].dt.tz_convert("UTC")
        else:
            # Otherwise, timestamps are returned as local, naive
            expected["A"] = expected["A"].dt.tz_localize(None)

        result = sql.read_sql_table("test_datetime_tz", self.conn)
        tm.assert_frame_equal(result, expected)

        result = sql.read_sql_query("SELECT * FROM test_datetime_tz", self.conn)
        if self.flavor == "sqlite":
            # read_sql_query does not return datetime type like read_sql_table
            assert isinstance(result.loc[0, "A"], str)
            result["A"] = to_datetime(result["A"])
        tm.assert_frame_equal(result, expected)

    def test_out_of_bounds_datetime(self):
        # GH 26761
        data = DataFrame({"date": datetime(9999, 1, 1)}, index=[0])
        assert data.to_sql(name="test_datetime_obb", con=self.conn, index=False) == 1
        result = sql.read_sql_table("test_datetime_obb", self.conn)
        expected = DataFrame([pd.NaT], columns=["date"])
        tm.assert_frame_equal(result, expected)

    def test_naive_datetimeindex_roundtrip(self):
        # GH 23510
        # Ensure that a naive DatetimeIndex isn't converted to UTC
        dates = date_range("2018-01-01", periods=5, freq="6H")._with_freq(None)
        expected = DataFrame({"nums": range(5)}, index=dates)
        assert (
            expected.to_sql(name="foo_table", con=self.conn, index_label="info_date")
            == 5
        )
        result = sql.read_sql_table("foo_table", self.conn, index_col="info_date")
        # result index with gain a name from a set_index operation; expected
        tm.assert_frame_equal(result, expected, check_names=False)

    def test_date_parsing(self):
        # No Parsing
        df = sql.read_sql_table("types", self.conn)
        expected_type = object if self.flavor == "sqlite" else np.datetime64
        assert issubclass(df.DateCol.dtype.type, expected_type)

        df = sql.read_sql_table("types", self.conn, parse_dates=["DateCol"])
        assert issubclass(df.DateCol.dtype.type, np.datetime64)

        df = sql.read_sql_table(
            "types", self.conn, parse_dates={"DateCol": "%Y-%m-%d %H:%M:%S"}
        )
        assert issubclass(df.DateCol.dtype.type, np.datetime64)

        df = sql.read_sql_table(
            "types",
            self.conn,
            parse_dates={"DateCol": {"format": "%Y-%m-%d %H:%M:%S"}},
        )
        assert issubclass(df.DateCol.dtype.type, np.datetime64)

        df = sql.read_sql_table("types", self.conn, parse_dates=["IntDateCol"])
        assert issubclass(df.IntDateCol.dtype.type, np.datetime64)

        df = sql.read_sql_table("types", self.conn, parse_dates={"IntDateCol": "s"})
        assert issubclass(df.IntDateCol.dtype.type, np.datetime64)

        df = sql.read_sql_table(
            "types", self.conn, parse_dates={"IntDateCol": {"unit": "s"}}
        )
        assert issubclass(df.IntDateCol.dtype.type, np.datetime64)

    def test_datetime(self):
        df = DataFrame(
            {"A": date_range("2013-01-01 09:00:00", periods=3), "B": np.arange(3.0)}
        )
        assert df.to_sql(name="test_datetime", con=self.conn) == 3

        # with read_table -> type information from schema used
        result = sql.read_sql_table("test_datetime", self.conn)
        result = result.drop("index", axis=1)
        tm.assert_frame_equal(result, df)

        # with read_sql -> no type information -> sqlite has no native
        result = sql.read_sql_query("SELECT * FROM test_datetime", self.conn)
        result = result.drop("index", axis=1)
        if self.flavor == "sqlite":
            assert isinstance(result.loc[0, "A"], str)
            result["A"] = to_datetime(result["A"])
            tm.assert_frame_equal(result, df)
        else:
            tm.assert_frame_equal(result, df)

    def test_datetime_NaT(self):
        df = DataFrame(
            {"A": date_range("2013-01-01 09:00:00", periods=3), "B": np.arange(3.0)}
        )
        df.loc[1, "A"] = np.nan
        assert df.to_sql(name="test_datetime", con=self.conn, index=False) == 3

        # with read_table -> type information from schema used
        result = sql.read_sql_table("test_datetime", self.conn)
        tm.assert_frame_equal(result, df)

        # with read_sql -> no type information -> sqlite has no native
        result = sql.read_sql_query("SELECT * FROM test_datetime", self.conn)
        if self.flavor == "sqlite":
            assert isinstance(result.loc[0, "A"], str)
            result["A"] = to_datetime(result["A"], errors="coerce")
            tm.assert_frame_equal(result, df)
        else:
            tm.assert_frame_equal(result, df)

    def test_datetime_date(self):
        # test support for datetime.date
        df = DataFrame([date(2014, 1, 1), date(2014, 1, 2)], columns=["a"])
        assert df.to_sql(name="test_date", con=self.conn, index=False) == 2
        res = read_sql_table("test_date", self.conn)
        result = res["a"]
        expected = to_datetime(df["a"])
        # comes back as datetime64
        tm.assert_series_equal(result, expected)

    def test_datetime_time(self, sqlite_buildin):
        # test support for datetime.time
        df = DataFrame([time(9, 0, 0), time(9, 1, 30)], columns=["a"])
        assert df.to_sql(name="test_time", con=self.conn, index=False) == 2
        res = read_sql_table("test_time", self.conn)
        tm.assert_frame_equal(res, df)

        # GH8341
        # first, use the fallback to have the sqlite adapter put in place
        sqlite_conn = sqlite_buildin
        assert sql.to_sql(df, "test_time2", sqlite_conn, index=False) == 2
        res = sql.read_sql_query("SELECT * FROM test_time2", sqlite_conn)
        ref = df.map(lambda _: _.strftime("%H:%M:%S.%f"))
        tm.assert_frame_equal(ref, res)  # check if adapter is in place
        # then test if sqlalchemy is unaffected by the sqlite adapter
        assert sql.to_sql(df, "test_time3", self.conn, index=False) == 2
        if self.flavor == "sqlite":
            res = sql.read_sql_query("SELECT * FROM test_time3", self.conn)
            ref = df.map(lambda _: _.strftime("%H:%M:%S.%f"))
            tm.assert_frame_equal(ref, res)
        res = sql.read_sql_table("test_time3", self.conn)
        tm.assert_frame_equal(df, res)

    def test_mixed_dtype_insert(self):
        # see GH6509
        s1 = Series(2**25 + 1, dtype=np.int32)
        s2 = Series(0.0, dtype=np.float32)
        df = DataFrame({"s1": s1, "s2": s2})

        # write and read again
        assert df.to_sql(name="test_read_write", con=self.conn, index=False) == 1
        df2 = sql.read_sql_table("test_read_write", self.conn)

        tm.assert_frame_equal(df, df2, check_dtype=False, check_exact=True)

    def test_nan_numeric(self):
        # NaNs in numeric float column
        df = DataFrame({"A": [0, 1, 2], "B": [0.2, np.nan, 5.6]})
        assert df.to_sql(name="test_nan", con=self.conn, index=False) == 3

        # with read_table
        result = sql.read_sql_table("test_nan", self.conn)
        tm.assert_frame_equal(result, df)

        # with read_sql
        result = sql.read_sql_query("SELECT * FROM test_nan", self.conn)
        tm.assert_frame_equal(result, df)

    def test_nan_fullcolumn(self):
        # full NaN column (numeric float column)
        df = DataFrame({"A": [0, 1, 2], "B": [np.nan, np.nan, np.nan]})
        assert df.to_sql(name="test_nan", con=self.conn, index=False) == 3

        # with read_table
        result = sql.read_sql_table("test_nan", self.conn)
        tm.assert_frame_equal(result, df)

        # with read_sql -> not type info from table -> stays None
        df["B"] = df["B"].astype("object")
        df["B"] = None
        result = sql.read_sql_query("SELECT * FROM test_nan", self.conn)
        tm.assert_frame_equal(result, df)

    def test_nan_string(self):
        # NaNs in string column
        df = DataFrame({"A": [0, 1, 2], "B": ["a", "b", np.nan]})
        assert df.to_sql(name="test_nan", con=self.conn, index=False) == 3

        # NaNs are coming back as None
        df.loc[2, "B"] = None

        # with read_table
        result = sql.read_sql_table("test_nan", self.conn)
        tm.assert_frame_equal(result, df)

        # with read_sql
        result = sql.read_sql_query("SELECT * FROM test_nan", self.conn)
        tm.assert_frame_equal(result, df)

    def _get_index_columns(self, tbl_name):
        from sqlalchemy import inspect

        insp = inspect(self.conn)

        ixs = insp.get_indexes(tbl_name)
        ixs = [i["column_names"] for i in ixs]
        return ixs

    def test_to_sql_save_index(self):
        self._to_sql_save_index()

    def test_transactions(self):
        self._transaction_test()

    def test_get_schema_create_table(self, test_frame3):
        # Use a dataframe without a bool column, since MySQL converts bool to
        # TINYINT (which read_sql_table returns as an int and causes a dtype
        # mismatch)
        from sqlalchemy import text
        from sqlalchemy.engine import Engine

        tbl = "test_get_schema_create_table"
        create_sql = sql.get_schema(test_frame3, tbl, con=self.conn)
        blank_test_df = test_frame3.iloc[:0]

        self.drop_table(tbl, self.conn)
        create_sql = text(create_sql)
        if isinstance(self.conn, Engine):
            with self.conn.connect() as conn:
                with conn.begin():
                    conn.execute(create_sql)
        else:
            with self.conn.begin():
                self.conn.execute(create_sql)
        returned_df = sql.read_sql_table(tbl, self.conn)
        tm.assert_frame_equal(returned_df, blank_test_df, check_index_type=False)
        self.drop_table(tbl, self.conn)

    def test_dtype(self):
        from sqlalchemy import (
            TEXT,
            String,
        )
        from sqlalchemy.schema import MetaData

        cols = ["A", "B"]
        data = [(0.8, True), (0.9, None)]
        df = DataFrame(data, columns=cols)
        assert df.to_sql(name="dtype_test", con=self.conn) == 2
        assert df.to_sql(name="dtype_test2", con=self.conn, dtype={"B": TEXT}) == 2
        meta = MetaData()
        meta.reflect(bind=self.conn)
        sqltype = meta.tables["dtype_test2"].columns["B"].type
        assert isinstance(sqltype, TEXT)
        msg = "The type of B is not a SQLAlchemy type"
        with pytest.raises(ValueError, match=msg):
            df.to_sql(name="error", con=self.conn, dtype={"B": str})

        # GH9083
        assert (
            df.to_sql(name="dtype_test3", con=self.conn, dtype={"B": String(10)}) == 2
        )
        meta.reflect(bind=self.conn)
        sqltype = meta.tables["dtype_test3"].columns["B"].type
        assert isinstance(sqltype, String)
        assert sqltype.length == 10

        # single dtype
        assert df.to_sql(name="single_dtype_test", con=self.conn, dtype=TEXT) == 2
        meta.reflect(bind=self.conn)
        sqltypea = meta.tables["single_dtype_test"].columns["A"].type
        sqltypeb = meta.tables["single_dtype_test"].columns["B"].type
        assert isinstance(sqltypea, TEXT)
        assert isinstance(sqltypeb, TEXT)

    def test_notna_dtype(self):
        from sqlalchemy import (
            Boolean,
            DateTime,
            Float,
            Integer,
        )
        from sqlalchemy.schema import MetaData

        cols = {
            "Bool": Series([True, None]),
            "Date": Series([datetime(2012, 5, 1), None]),
            "Int": Series([1, None], dtype="object"),
            "Float": Series([1.1, None]),
        }
        df = DataFrame(cols)

        tbl = "notna_dtype_test"
        assert df.to_sql(name=tbl, con=self.conn) == 2
        _ = sql.read_sql_table(tbl, self.conn)
        meta = MetaData()
        meta.reflect(bind=self.conn)
        my_type = Integer if self.flavor == "mysql" else Boolean
        col_dict = meta.tables[tbl].columns
        assert isinstance(col_dict["Bool"].type, my_type)
        assert isinstance(col_dict["Date"].type, DateTime)
        assert isinstance(col_dict["Int"].type, Integer)
        assert isinstance(col_dict["Float"].type, Float)

    def test_double_precision(self):
        from sqlalchemy import (
            BigInteger,
            Float,
            Integer,
        )
        from sqlalchemy.schema import MetaData

        V = 1.23456789101112131415

        df = DataFrame(
            {
                "f32": Series([V], dtype="float32"),
                "f64": Series([V], dtype="float64"),
                "f64_as_f32": Series([V], dtype="float64"),
                "i32": Series([5], dtype="int32"),
                "i64": Series([5], dtype="int64"),
            }
        )

        assert (
            df.to_sql(
                name="test_dtypes",
                con=self.conn,
                index=False,
                if_exists="replace",
                dtype={"f64_as_f32": Float(precision=23)},
            )
            == 1
        )
        res = sql.read_sql_table("test_dtypes", self.conn)

        # check precision of float64
        assert np.round(df["f64"].iloc[0], 14) == np.round(res["f64"].iloc[0], 14)

        # check sql types
        meta = MetaData()
        meta.reflect(bind=self.conn)
        col_dict = meta.tables["test_dtypes"].columns
        assert str(col_dict["f32"].type) == str(col_dict["f64_as_f32"].type)
        assert isinstance(col_dict["f32"].type, Float)
        assert isinstance(col_dict["f64"].type, Float)
        assert isinstance(col_dict["i32"].type, Integer)
        assert isinstance(col_dict["i64"].type, BigInteger)

    def test_connectable_issue_example(self):
        # This tests the example raised in issue
        # https://github.com/pandas-dev/pandas/issues/10104
        from sqlalchemy.engine import Engine

        def test_select(connection):
            query = "SELECT test_foo_data FROM test_foo_data"
            return sql.read_sql_query(query, con=connection)

        def test_append(connection, data):
            data.to_sql(name="test_foo_data", con=connection, if_exists="append")

        def test_connectable(conn):
            # https://github.com/sqlalchemy/sqlalchemy/commit/
            # 00b5c10846e800304caa86549ab9da373b42fa5d#r48323973
            foo_data = test_select(conn)
            test_append(conn, foo_data)

        def main(connectable):
            if isinstance(connectable, Engine):
                with connectable.connect() as conn:
                    with conn.begin():
                        test_connectable(conn)
            else:
                test_connectable(connectable)

        assert (
            DataFrame({"test_foo_data": [0, 1, 2]}).to_sql(
                name="test_foo_data", con=self.conn
            )
            == 3
        )
        main(self.conn)

    @pytest.mark.parametrize(
        "input",
        [{"foo": [np.inf]}, {"foo": [-np.inf]}, {"foo": [-np.inf], "infe0": ["bar"]}],
    )
    def test_to_sql_with_negative_npinf(self, input, request):
        # GH 34431

        df = DataFrame(input)

        if self.flavor == "mysql":
            # GH 36465
            # The input {"foo": [-np.inf], "infe0": ["bar"]} does not raise any error
            # for pymysql version >= 0.10
            # TODO(GH#36465): remove this version check after GH 36465 is fixed
            pymysql = pytest.importorskip("pymysql")

            if (
                Version(pymysql.__version__) < Version("1.0.3")
                and "infe0" in df.columns
            ):
                mark = pytest.mark.xfail(reason="GH 36465")
                request.node.add_marker(mark)

            msg = "inf cannot be used with MySQL"
            with pytest.raises(ValueError, match=msg):
                df.to_sql(name="foobar", con=self.conn, index=False)
        else:
            assert df.to_sql(name="foobar", con=self.conn, index=False) == 1
            res = sql.read_sql_table("foobar", self.conn)
            tm.assert_equal(df, res)

    def test_temporary_table(self):
        from sqlalchemy import (
            Column,
            Integer,
            Unicode,
            select,
        )
        from sqlalchemy.orm import (
            Session,
            declarative_base,
        )

        test_data = "Hello, World!"
        expected = DataFrame({"spam": [test_data]})
        Base = declarative_base()

        class Temporary(Base):
            __tablename__ = "temp_test"
            __table_args__ = {"prefixes": ["TEMPORARY"]}
            id = Column(Integer, primary_key=True)
            spam = Column(Unicode(30), nullable=False)

        with Session(self.conn) as session:
            with session.begin():
                conn = session.connection()
                Temporary.__table__.create(conn)
                session.add(Temporary(spam=test_data))
                session.flush()
                df = sql.read_sql_query(sql=select(Temporary.spam), con=conn)
        tm.assert_frame_equal(df, expected)

    # -- SQL Engine tests (in the base class for now)
    def test_invalid_engine(self, test_frame1):
        msg = "engine must be one of 'auto', 'sqlalchemy'"
        with pytest.raises(ValueError, match=msg):
            self._to_sql_with_sql_engine(test_frame1, "bad_engine")

    def test_options_sqlalchemy(self, test_frame1):
        # use the set option
        with pd.option_context("io.sql.engine", "sqlalchemy"):
            self._to_sql_with_sql_engine(test_frame1)

    def test_options_auto(self, test_frame1):
        # use the set option
        with pd.option_context("io.sql.engine", "auto"):
            self._to_sql_with_sql_engine(test_frame1)

    def test_options_get_engine(self):
        assert isinstance(get_engine("sqlalchemy"), SQLAlchemyEngine)

        with pd.option_context("io.sql.engine", "sqlalchemy"):
            assert isinstance(get_engine("auto"), SQLAlchemyEngine)
            assert isinstance(get_engine("sqlalchemy"), SQLAlchemyEngine)

        with pd.option_context("io.sql.engine", "auto"):
            assert isinstance(get_engine("auto"), SQLAlchemyEngine)
            assert isinstance(get_engine("sqlalchemy"), SQLAlchemyEngine)

    def test_get_engine_auto_error_message(self):
        # Expect different error messages from get_engine(engine="auto")
        # if engines aren't installed vs. are installed but bad version
        pass
        # TODO(GH#36893) fill this in when we add more engines

    @pytest.mark.parametrize("func", ["read_sql", "read_sql_query"])
    def test_read_sql_dtype_backend(self, string_storage, func, dtype_backend):
        # GH#50048
        table = "test"
        df = self.dtype_backend_data()
        df.to_sql(name=table, con=self.conn, index=False, if_exists="replace")

        with pd.option_context("mode.string_storage", string_storage):
            result = getattr(pd, func)(
                f"Select * from {table}", self.conn, dtype_backend=dtype_backend
            )
        expected = self.dtype_backend_expected(string_storage, dtype_backend)
        tm.assert_frame_equal(result, expected)

        with pd.option_context("mode.string_storage", string_storage):
            iterator = getattr(pd, func)(
                f"Select * from {table}",
                con=self.conn,
                dtype_backend=dtype_backend,
                chunksize=3,
            )
            expected = self.dtype_backend_expected(string_storage, dtype_backend)
            for result in iterator:
                tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("func", ["read_sql", "read_sql_table"])
    def test_read_sql_dtype_backend_table(self, string_storage, func, dtype_backend):
        # GH#50048
        table = "test"
        df = self.dtype_backend_data()
        df.to_sql(name=table, con=self.conn, index=False, if_exists="replace")

        with pd.option_context("mode.string_storage", string_storage):
            result = getattr(pd, func)(table, self.conn, dtype_backend=dtype_backend)
        expected = self.dtype_backend_expected(string_storage, dtype_backend)
        tm.assert_frame_equal(result, expected)

        with pd.option_context("mode.string_storage", string_storage):
            iterator = getattr(pd, func)(
                table,
                self.conn,
                dtype_backend=dtype_backend,
                chunksize=3,
            )
            expected = self.dtype_backend_expected(string_storage, dtype_backend)
            for result in iterator:
                tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("func", ["read_sql", "read_sql_table", "read_sql_query"])
    def test_read_sql_invalid_dtype_backend_table(self, func):
        table = "test"
        df = self.dtype_backend_data()
        df.to_sql(name=table, con=self.conn, index=False, if_exists="replace")

        msg = (
            "dtype_backend numpy is invalid, only 'numpy_nullable' and "
            "'pyarrow' are allowed."
        )
        with pytest.raises(ValueError, match=msg):
            getattr(pd, func)(table, self.conn, dtype_backend="numpy")

    def dtype_backend_data(self) -> DataFrame:
        return DataFrame(
            {
                "a": Series([1, np.nan, 3], dtype="Int64"),
                "b": Series([1, 2, 3], dtype="Int64"),
                "c": Series([1.5, np.nan, 2.5], dtype="Float64"),
                "d": Series([1.5, 2.0, 2.5], dtype="Float64"),
                "e": [True, False, None],
                "f": [True, False, True],
                "g": ["a", "b", "c"],
                "h": ["a", "b", None],
            }
        )

    def dtype_backend_expected(self, storage, dtype_backend) -> DataFrame:
        string_array: StringArray | ArrowStringArray
        string_array_na: StringArray | ArrowStringArray
        if storage == "python":
            string_array = StringArray(np.array(["a", "b", "c"], dtype=np.object_))
            string_array_na = StringArray(np.array(["a", "b", pd.NA], dtype=np.object_))

        else:
            pa = pytest.importorskip("pyarrow")
            string_array = ArrowStringArray(pa.array(["a", "b", "c"]))
            string_array_na = ArrowStringArray(pa.array(["a", "b", None]))

        df = DataFrame(
            {
                "a": Series([1, np.nan, 3], dtype="Int64"),
                "b": Series([1, 2, 3], dtype="Int64"),
                "c": Series([1.5, np.nan, 2.5], dtype="Float64"),
                "d": Series([1.5, 2.0, 2.5], dtype="Float64"),
                "e": Series([True, False, pd.NA], dtype="boolean"),
                "f": Series([True, False, True], dtype="boolean"),
                "g": string_array,
                "h": string_array_na,
            }
        )
        if dtype_backend == "pyarrow":
            pa = pytest.importorskip("pyarrow")

            from pandas.arrays import ArrowExtensionArray

            df = DataFrame(
                {
                    col: ArrowExtensionArray(pa.array(df[col], from_pandas=True))
                    for col in df.columns
                }
            )
        return df

    def test_chunksize_empty_dtypes(self):
        # GH#50245
        dtypes = {"a": "int64", "b": "object"}
        df = DataFrame(columns=["a", "b"]).astype(dtypes)
        expected = df.copy()
        df.to_sql(name="test", con=self.conn, index=False, if_exists="replace")

        for result in read_sql_query(
            "SELECT * FROM test",
            self.conn,
            dtype=dtypes,
            chunksize=1,
        ):
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype_backend", [lib.no_default, "numpy_nullable"])
    @pytest.mark.parametrize("func", ["read_sql", "read_sql_query"])
    def test_read_sql_dtype(self, func, dtype_backend):
        # GH#50797
        table = "test"
        df = DataFrame({"a": [1, 2, 3], "b": 5})
        df.to_sql(name=table, con=self.conn, index=False, if_exists="replace")

        result = getattr(pd, func)(
            f"Select * from {table}",
            self.conn,
            dtype={"a": np.float64},
            dtype_backend=dtype_backend,
        )
        expected = DataFrame(
            {
                "a": Series([1, 2, 3], dtype=np.float64),
                "b": Series(
                    [5, 5, 5],
                    dtype="int64" if not dtype_backend == "numpy_nullable" else "Int64",
                ),
            }
        )
        tm.assert_frame_equal(result, expected)


class TestSQLiteAlchemy(_TestSQLAlchemy):
    """
    Test the sqlalchemy backend against an in-memory sqlite database.

    """

    flavor = "sqlite"

    @classmethod
    def setup_engine(cls):
        cls.engine = sqlalchemy.create_engine("sqlite:///:memory:")

    @classmethod
    def setup_driver(cls):
        # sqlite3 is built-in
        cls.driver = None

    def test_keyword_deprecation(self):
        # GH 54397
        msg = (
            "Starting with pandas version 3.0 all arguments of to_sql except for the "
            "arguments 'name' and 'con' will be keyword-only."
        )
        df = DataFrame([{"A": 1, "B": 2, "C": 3}, {"A": 1, "B": 2, "C": 3}])
        df.to_sql("example", self.conn)

        with tm.assert_produces_warning(FutureWarning, match=msg):
            df.to_sql("example", self.conn, None, if_exists="replace")

    def test_default_type_conversion(self):
        df = sql.read_sql_table("types", self.conn)

        assert issubclass(df.FloatCol.dtype.type, np.floating)
        assert issubclass(df.IntCol.dtype.type, np.integer)

        # sqlite has no boolean type, so integer type is returned
        assert issubclass(df.BoolCol.dtype.type, np.integer)

        # Int column with NA values stays as float
        assert issubclass(df.IntColWithNull.dtype.type, np.floating)

        # Non-native Bool column with NA values stays as float
        assert issubclass(df.BoolColWithNull.dtype.type, np.floating)

    def test_default_date_load(self):
        df = sql.read_sql_table("types", self.conn)

        # IMPORTANT - sqlite has no native date type, so shouldn't parse, but
        assert not issubclass(df.DateCol.dtype.type, np.datetime64)

    def test_bigint_warning(self):
        # test no warning for BIGINT (to support int64) is raised (GH7433)
        df = DataFrame({"a": [1, 2]}, dtype="int64")
        assert df.to_sql(name="test_bigintwarning", con=self.conn, index=False) == 2

        with tm.assert_produces_warning(None):
            sql.read_sql_table("test_bigintwarning", self.conn)

    def test_valueerror_exception(self):
        df = DataFrame({"col1": [1, 2], "col2": [3, 4]})
        with pytest.raises(ValueError, match="Empty table name specified"):
            df.to_sql(name="", con=self.conn, if_exists="replace", index=False)

    def test_row_object_is_named_tuple(self):
        # GH 40682
        # Test for the is_named_tuple() function
        # Placed here due to its usage of sqlalchemy

        from sqlalchemy import (
            Column,
            Integer,
            String,
        )
        from sqlalchemy.orm import (
            declarative_base,
            sessionmaker,
        )

        BaseModel = declarative_base()

        class Test(BaseModel):
            __tablename__ = "test_frame"
            id = Column(Integer, primary_key=True)
            string_column = Column(String(50))

        with self.conn.begin():
            BaseModel.metadata.create_all(self.conn)
        Session = sessionmaker(bind=self.conn)
        with Session() as session:
            df = DataFrame({"id": [0, 1], "string_column": ["hello", "world"]})
            assert (
                df.to_sql(
                    name="test_frame", con=self.conn, index=False, if_exists="replace"
                )
                == 2
            )
            session.commit()
            test_query = session.query(Test.id, Test.string_column)
            df = DataFrame(test_query)

        assert list(df.columns) == ["id", "string_column"]

    def dtype_backend_expected(self, storage, dtype_backend) -> DataFrame:
        df = super().dtype_backend_expected(storage, dtype_backend)
        if dtype_backend == "numpy_nullable":
            df = df.astype({"e": "Int64", "f": "Int64"})
        else:
            df = df.astype({"e": "int64[pyarrow]", "f": "int64[pyarrow]"})

        return df

    @pytest.mark.parametrize("func", ["read_sql", "read_sql_table"])
    def test_read_sql_dtype_backend_table(self, string_storage, func):
        # GH#50048 Not supported for sqlite
        pass

    def test_read_sql_string_inference(self):
        # GH#54430
        pytest.importorskip("pyarrow")
        table = "test"
        df = DataFrame({"a": ["x", "y"]})
        df.to_sql(table, con=self.conn, index=False, if_exists="replace")

        with pd.option_context("future.infer_string", True):
            result = read_sql_table(table, self.conn)

        dtype = "string[pyarrow_numpy]"
        expected = DataFrame(
            {"a": ["x", "y"]}, dtype=dtype, columns=Index(["a"], dtype=dtype)
        )

        tm.assert_frame_equal(result, expected)


@pytest.mark.db
class TestMySQLAlchemy(_TestSQLAlchemy):
    """
    Test the sqlalchemy backend against an MySQL database.

    """

    flavor = "mysql"
    port = 3306

    @classmethod
    def setup_engine(cls):
        cls.engine = sqlalchemy.create_engine(
            f"mysql+{cls.driver}://root@localhost:{cls.port}/pandas",
            connect_args=cls.connect_args,
        )

    @classmethod
    def setup_driver(cls):
        pymysql = pytest.importorskip("pymysql")
        cls.driver = "pymysql"
        cls.connect_args = {"client_flag": pymysql.constants.CLIENT.MULTI_STATEMENTS}

    def test_default_type_conversion(self):
        pass

    def dtype_backend_expected(self, storage, dtype_backend) -> DataFrame:
        df = super().dtype_backend_expected(storage, dtype_backend)
        if dtype_backend == "numpy_nullable":
            df = df.astype({"e": "Int64", "f": "Int64"})
        else:
            df = df.astype({"e": "int64[pyarrow]", "f": "int64[pyarrow]"})

        return df


@pytest.mark.db
class TestPostgreSQLAlchemy(_TestSQLAlchemy):
    """
    Test the sqlalchemy backend against an PostgreSQL database.

    """

    flavor = "postgresql"
    port = 5432

    @classmethod
    def setup_engine(cls):
        cls.engine = sqlalchemy.create_engine(
            f"postgresql+{cls.driver}://postgres:postgres@localhost:{cls.port}/pandas"
        )

    @classmethod
    def setup_driver(cls):
        pytest.importorskip("psycopg2")
        cls.driver = "psycopg2"

    def test_schema_support(self):
        from sqlalchemy.engine import Engine

        # only test this for postgresql (schema's not supported in
        # mysql/sqlite)
        df = DataFrame({"col1": [1, 2], "col2": [0.1, 0.2], "col3": ["a", "n"]})

        # create a schema
        with self.conn.begin():
            self.conn.exec_driver_sql("DROP SCHEMA IF EXISTS other CASCADE;")
            self.conn.exec_driver_sql("CREATE SCHEMA other;")

        # write dataframe to different schema's
        assert df.to_sql(name="test_schema_public", con=self.conn, index=False) == 2
        assert (
            df.to_sql(
                name="test_schema_public_explicit",
                con=self.conn,
                index=False,
                schema="public",
            )
            == 2
        )
        assert (
            df.to_sql(
                name="test_schema_other", con=self.conn, index=False, schema="other"
            )
            == 2
        )

        # read dataframes back in
        res1 = sql.read_sql_table("test_schema_public", self.conn)
        tm.assert_frame_equal(df, res1)
        res2 = sql.read_sql_table("test_schema_public_explicit", self.conn)
        tm.assert_frame_equal(df, res2)
        res3 = sql.read_sql_table(
            "test_schema_public_explicit", self.conn, schema="public"
        )
        tm.assert_frame_equal(df, res3)
        res4 = sql.read_sql_table("test_schema_other", self.conn, schema="other")
        tm.assert_frame_equal(df, res4)
        msg = "Table test_schema_other not found"
        with pytest.raises(ValueError, match=msg):
            sql.read_sql_table("test_schema_other", self.conn, schema="public")

        # different if_exists options

        # create a schema
        with self.conn.begin():
            self.conn.exec_driver_sql("DROP SCHEMA IF EXISTS other CASCADE;")
            self.conn.exec_driver_sql("CREATE SCHEMA other;")

        # write dataframe with different if_exists options
        assert (
            df.to_sql(
                name="test_schema_other", con=self.conn, schema="other", index=False
            )
            == 2
        )
        df.to_sql(
            name="test_schema_other",
            con=self.conn,
            schema="other",
            index=False,
            if_exists="replace",
        )
        assert (
            df.to_sql(
                name="test_schema_other",
                con=self.conn,
                schema="other",
                index=False,
                if_exists="append",
            )
            == 2
        )
        res = sql.read_sql_table("test_schema_other", self.conn, schema="other")
        tm.assert_frame_equal(concat([df, df], ignore_index=True), res)

        # specifying schema in user-provided meta

        # The schema won't be applied on another Connection
        # because of transactional schemas
        if isinstance(self.conn, Engine):
            engine2 = self.connect()
            pdsql = sql.SQLDatabase(engine2, schema="other")
            assert pdsql.to_sql(df, "test_schema_other2", index=False) == 2
            assert (
                pdsql.to_sql(df, "test_schema_other2", index=False, if_exists="replace")
                == 2
            )
            assert (
                pdsql.to_sql(df, "test_schema_other2", index=False, if_exists="append")
                == 2
            )
            res1 = sql.read_sql_table("test_schema_other2", self.conn, schema="other")
            res2 = pdsql.read_table("test_schema_other2")
            tm.assert_frame_equal(res1, res2)

    def test_self_join_date_columns(self):
        # GH 44421
        from sqlalchemy.engine import Engine
        from sqlalchemy.sql import text

        create_table = text(
            """
        CREATE TABLE person
        (
            id serial constraint person_pkey primary key,
            created_dt timestamp with time zone
        );

        INSERT INTO person
            VALUES (1, '2021-01-01T00:00:00Z');
        """
        )
        if isinstance(self.conn, Engine):
            with self.conn.connect() as con:
                with con.begin():
                    con.execute(create_table)
        else:
            with self.conn.begin():
                self.conn.execute(create_table)

        sql_query = (
            'SELECT * FROM "person" AS p1 INNER JOIN "person" AS p2 ON p1.id = p2.id;'
        )
        result = pd.read_sql(sql_query, self.conn)
        expected = DataFrame(
            [[1, Timestamp("2021", tz="UTC")] * 2], columns=["id", "created_dt"] * 2
        )
        tm.assert_frame_equal(result, expected)

        # Cleanup
        with sql.SQLDatabase(self.conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("person")


# -----------------------------------------------------------------------------
# -- Test Sqlite / MySQL fallback


class TestSQLiteFallback(SQLiteMixIn, PandasSQLTest):
    """
    Test the fallback mode against an in-memory sqlite database.

    """

    flavor = "sqlite"

    @pytest.fixture(autouse=True)
    def setup_method(self, iris_path, types_data):
        self.conn = self.connect()
        self.load_iris_data(iris_path)
        self.load_types_data(types_data)
        self.pandasSQL = sql.SQLiteDatabase(self.conn)

    def test_read_sql_parameter(self, sql_strings):
        self._read_sql_iris_parameter(sql_strings)

    def test_read_sql_named_parameter(self, sql_strings):
        self._read_sql_iris_named_parameter(sql_strings)

    def test_to_sql_empty(self, test_frame1):
        self._to_sql_empty(test_frame1)

    def test_create_and_drop_table(self):
        temp_frame = DataFrame(
            {"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]}
        )

        assert self.pandasSQL.to_sql(temp_frame, "drop_test_frame") == 4

        assert self.pandasSQL.has_table("drop_test_frame")

        self.pandasSQL.drop_table("drop_test_frame")

        assert not self.pandasSQL.has_table("drop_test_frame")

    def test_roundtrip(self, test_frame1):
        self._roundtrip(test_frame1)

    def test_execute_sql(self):
        self._execute_sql()

    def test_datetime_date(self):
        # test support for datetime.date
        df = DataFrame([date(2014, 1, 1), date(2014, 1, 2)], columns=["a"])
        assert df.to_sql(name="test_date", con=self.conn, index=False) == 2
        res = read_sql_query("SELECT * FROM test_date", self.conn)
        if self.flavor == "sqlite":
            # comes back as strings
            tm.assert_frame_equal(res, df.astype(str))
        elif self.flavor == "mysql":
            tm.assert_frame_equal(res, df)

    @pytest.mark.parametrize("tz_aware", [False, True])
    def test_datetime_time(self, tz_aware):
        # test support for datetime.time, GH #8341
        if not tz_aware:
            tz_times = [time(9, 0, 0), time(9, 1, 30)]
        else:
            tz_dt = date_range("2013-01-01 09:00:00", periods=2, tz="US/Pacific")
            tz_times = Series(tz_dt.to_pydatetime()).map(lambda dt: dt.timetz())

        df = DataFrame(tz_times, columns=["a"])

        assert df.to_sql(name="test_time", con=self.conn, index=False) == 2
        res = read_sql_query("SELECT * FROM test_time", self.conn)
        if self.flavor == "sqlite":
            # comes back as strings
            expected = df.map(lambda _: _.strftime("%H:%M:%S.%f"))
            tm.assert_frame_equal(res, expected)

    def _get_index_columns(self, tbl_name):
        ixs = sql.read_sql_query(
            "SELECT * FROM sqlite_master WHERE type = 'index' "
            f"AND tbl_name = '{tbl_name}'",
            self.conn,
        )
        ix_cols = []
        for ix_name in ixs.name:
            ix_info = sql.read_sql_query(f"PRAGMA index_info({ix_name})", self.conn)
            ix_cols.append(ix_info.name.tolist())
        return ix_cols

    def test_to_sql_save_index(self):
        self._to_sql_save_index()

    def test_transactions(self):
        self._transaction_test()

    def _get_sqlite_column_type(self, table, column):
        recs = self.conn.execute(f"PRAGMA table_info({table})")
        for cid, name, ctype, not_null, default, pk in recs:
            if name == column:
                return ctype
        raise ValueError(f"Table {table}, column {column} not found")

    def test_dtype(self):
        if self.flavor == "mysql":
            pytest.skip("Not applicable to MySQL legacy")
        cols = ["A", "B"]
        data = [(0.8, True), (0.9, None)]
        df = DataFrame(data, columns=cols)
        assert df.to_sql(name="dtype_test", con=self.conn) == 2
        assert df.to_sql(name="dtype_test2", con=self.conn, dtype={"B": "STRING"}) == 2

        # sqlite stores Boolean values as INTEGER
        assert self._get_sqlite_column_type("dtype_test", "B") == "INTEGER"

        assert self._get_sqlite_column_type("dtype_test2", "B") == "STRING"
        msg = r"B \(<class 'bool'>\) not a string"
        with pytest.raises(ValueError, match=msg):
            df.to_sql(name="error", con=self.conn, dtype={"B": bool})

        # single dtype
        assert df.to_sql(name="single_dtype_test", con=self.conn, dtype="STRING") == 2
        assert self._get_sqlite_column_type("single_dtype_test", "A") == "STRING"
        assert self._get_sqlite_column_type("single_dtype_test", "B") == "STRING"

    def test_notna_dtype(self):
        if self.flavor == "mysql":
            pytest.skip("Not applicable to MySQL legacy")

        cols = {
            "Bool": Series([True, None]),
            "Date": Series([datetime(2012, 5, 1), None]),
            "Int": Series([1, None], dtype="object"),
            "Float": Series([1.1, None]),
        }
        df = DataFrame(cols)

        tbl = "notna_dtype_test"
        assert df.to_sql(name=tbl, con=self.conn) == 2

        assert self._get_sqlite_column_type(tbl, "Bool") == "INTEGER"
        assert self._get_sqlite_column_type(tbl, "Date") == "TIMESTAMP"
        assert self._get_sqlite_column_type(tbl, "Int") == "INTEGER"
        assert self._get_sqlite_column_type(tbl, "Float") == "REAL"

    def test_illegal_names(self):
        # For sqlite, these should work fine
        df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])

        msg = "Empty table or column name specified"
        with pytest.raises(ValueError, match=msg):
            df.to_sql(name="", con=self.conn)

        for ndx, weird_name in enumerate(
            [
                "test_weird_name]",
                "test_weird_name[",
                "test_weird_name`",
                'test_weird_name"',
                "test_weird_name'",
                "_b.test_weird_name_01-30",
                '"_b.test_weird_name_01-30"',
                "99beginswithnumber",
                "12345",
                "\xe9",
            ]
        ):
            assert df.to_sql(name=weird_name, con=self.conn) == 2
            sql.table_exists(weird_name, self.conn)

            df2 = DataFrame([[1, 2], [3, 4]], columns=["a", weird_name])
            c_tbl = f"test_weird_col_name{ndx:d}"
            assert df2.to_sql(name=c_tbl, con=self.conn) == 2
            sql.table_exists(c_tbl, self.conn)


# -----------------------------------------------------------------------------
# -- Old tests from 0.13.1 (before refactor using sqlalchemy)


_formatters = {
    datetime: "'{}'".format,
    str: "'{}'".format,
    np.str_: "'{}'".format,
    bytes: "'{}'".format,
    float: "{:.8f}".format,
    int: "{:d}".format,
    type(None): lambda x: "NULL",
    np.float64: "{:.10f}".format,
    bool: "'{!s}'".format,
}


def format_query(sql, *args):
    processed_args = []
    for arg in args:
        if isinstance(arg, float) and isna(arg):
            arg = None

        formatter = _formatters[type(arg)]
        processed_args.append(formatter(arg))

    return sql % tuple(processed_args)


def tquery(query, con=None):
    """Replace removed sql.tquery function"""
    with sql.pandasSQL_builder(con) as pandas_sql:
        res = pandas_sql.execute(query).fetchall()
    return None if res is None else list(res)


class TestXSQLite:
    def drop_table(self, table_name, conn):
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {sql._get_valid_sqlite_name(table_name)}")
        conn.commit()

    def test_basic(self, sqlite_buildin):
        frame = tm.makeTimeDataFrame()
        assert (
            sql.to_sql(frame, name="test_table", con=sqlite_buildin, index=False) == 30
        )
        result = sql.read_sql("select * from test_table", sqlite_buildin)

        # HACK! Change this once indexes are handled properly.
        result.index = frame.index

        expected = frame
        tm.assert_frame_equal(result, frame)

        frame["txt"] = ["a"] * len(frame)
        frame2 = frame.copy()
        new_idx = Index(np.arange(len(frame2)), dtype=np.int64) + 10
        frame2["Idx"] = new_idx.copy()
        assert (
            sql.to_sql(frame2, name="test_table2", con=sqlite_buildin, index=False)
            == 30
        )
        result = sql.read_sql(
            "select * from test_table2", sqlite_buildin, index_col="Idx"
        )
        expected = frame.copy()
        expected.index = new_idx
        expected.index.name = "Idx"
        tm.assert_frame_equal(expected, result)

    def test_write_row_by_row(self, sqlite_buildin):
        frame = tm.makeTimeDataFrame()
        frame.iloc[0, 0] = np.nan
        create_sql = sql.get_schema(frame, "test")
        cur = sqlite_buildin.cursor()
        cur.execute(create_sql)

        ins = "INSERT INTO test VALUES (%s, %s, %s, %s)"
        for _, row in frame.iterrows():
            fmt_sql = format_query(ins, *row)
            tquery(fmt_sql, con=sqlite_buildin)

        sqlite_buildin.commit()

        result = sql.read_sql("select * from test", con=sqlite_buildin)
        result.index = frame.index
        tm.assert_frame_equal(result, frame, rtol=1e-3)

    def test_execute(self, sqlite_buildin):
        frame = tm.makeTimeDataFrame()
        create_sql = sql.get_schema(frame, "test")
        cur = sqlite_buildin.cursor()
        cur.execute(create_sql)
        ins = "INSERT INTO test VALUES (?, ?, ?, ?)"

        row = frame.iloc[0]
        with sql.pandasSQL_builder(sqlite_buildin) as pandas_sql:
            pandas_sql.execute(ins, tuple(row))
        sqlite_buildin.commit()

        result = sql.read_sql("select * from test", sqlite_buildin)
        result.index = frame.index[:1]
        tm.assert_frame_equal(result, frame[:1])

    def test_schema(self, sqlite_buildin):
        frame = tm.makeTimeDataFrame()
        create_sql = sql.get_schema(frame, "test")
        lines = create_sql.splitlines()
        for line in lines:
            tokens = line.split(" ")
            if len(tokens) == 2 and tokens[0] == "A":
                assert tokens[1] == "DATETIME"

        create_sql = sql.get_schema(frame, "test", keys=["A", "B"])
        lines = create_sql.splitlines()
        assert 'PRIMARY KEY ("A", "B")' in create_sql
        cur = sqlite_buildin.cursor()
        cur.execute(create_sql)

    def test_execute_fail(self, sqlite_buildin):
        create_sql = """
        CREATE TABLE test
        (
        a TEXT,
        b TEXT,
        c REAL,
        PRIMARY KEY (a, b)
        );
        """
        cur = sqlite_buildin.cursor()
        cur.execute(create_sql)

        with sql.pandasSQL_builder(sqlite_buildin) as pandas_sql:
            pandas_sql.execute('INSERT INTO test VALUES("foo", "bar", 1.234)')
            pandas_sql.execute('INSERT INTO test VALUES("foo", "baz", 2.567)')

            with pytest.raises(sql.DatabaseError, match="Execution failed on sql"):
                pandas_sql.execute('INSERT INTO test VALUES("foo", "bar", 7)')

    def test_execute_closed_connection(self):
        create_sql = """
        CREATE TABLE test
        (
        a TEXT,
        b TEXT,
        c REAL,
        PRIMARY KEY (a, b)
        );
        """
        with contextlib.closing(sqlite3.connect(":memory:")) as conn:
            cur = conn.cursor()
            cur.execute(create_sql)

            with sql.pandasSQL_builder(conn) as pandas_sql:
                pandas_sql.execute('INSERT INTO test VALUES("foo", "bar", 1.234)')

        msg = "Cannot operate on a closed database."
        with pytest.raises(sqlite3.ProgrammingError, match=msg):
            tquery("select * from test", con=conn)

    def test_keyword_as_column_names(self, sqlite_buildin):
        df = DataFrame({"From": np.ones(5)})
        assert sql.to_sql(df, con=sqlite_buildin, name="testkeywords", index=False) == 5

    def test_onecolumn_of_integer(self, sqlite_buildin):
        # GH 3628
        # a column_of_integers dataframe should transfer well to sql

        mono_df = DataFrame([1, 2], columns=["c0"])
        assert sql.to_sql(mono_df, con=sqlite_buildin, name="mono_df", index=False) == 2
        # computing the sum via sql
        con_x = sqlite_buildin
        the_sum = sum(my_c0[0] for my_c0 in con_x.execute("select * from mono_df"))
        # it should not fail, and gives 3 ( Issue #3628 )
        assert the_sum == 3

        result = sql.read_sql("select * from mono_df", con_x)
        tm.assert_frame_equal(result, mono_df)

    def test_if_exists(self, sqlite_buildin):
        df_if_exists_1 = DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
        df_if_exists_2 = DataFrame({"col1": [3, 4, 5], "col2": ["C", "D", "E"]})
        table_name = "table_if_exists"
        sql_select = f"SELECT * FROM {table_name}"

        msg = "'notvalidvalue' is not valid for if_exists"
        with pytest.raises(ValueError, match=msg):
            sql.to_sql(
                frame=df_if_exists_1,
                con=sqlite_buildin,
                name=table_name,
                if_exists="notvalidvalue",
            )
        self.drop_table(table_name, sqlite_buildin)

        # test if_exists='fail'
        sql.to_sql(
            frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="fail"
        )
        msg = "Table 'table_if_exists' already exists"
        with pytest.raises(ValueError, match=msg):
            sql.to_sql(
                frame=df_if_exists_1,
                con=sqlite_buildin,
                name=table_name,
                if_exists="fail",
            )
        # test if_exists='replace'
        sql.to_sql(
            frame=df_if_exists_1,
            con=sqlite_buildin,
            name=table_name,
            if_exists="replace",
            index=False,
        )
        assert tquery(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B")]
        assert (
            sql.to_sql(
                frame=df_if_exists_2,
                con=sqlite_buildin,
                name=table_name,
                if_exists="replace",
                index=False,
            )
            == 3
        )
        assert tquery(sql_select, con=sqlite_buildin) == [(3, "C"), (4, "D"), (5, "E")]
        self.drop_table(table_name, sqlite_buildin)

        # test if_exists='append'
        assert (
            sql.to_sql(
                frame=df_if_exists_1,
                con=sqlite_buildin,
                name=table_name,
                if_exists="fail",
                index=False,
            )
            == 2
        )
        assert tquery(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B")]
        assert (
            sql.to_sql(
                frame=df_if_exists_2,
                con=sqlite_buildin,
                name=table_name,
                if_exists="append",
                index=False,
            )
            == 3
        )
        assert tquery(sql_select, con=sqlite_buildin) == [
            (1, "A"),
            (2, "B"),
            (3, "C"),
            (4, "D"),
            (5, "E"),
        ]
        self.drop_table(table_name, sqlite_buildin)
