import pyodbc
import pandas as pd
import os




def create_connection():
    server = '.\\SQLEXPRESS'
    database = 'MegaMoldInternationalDB'
    driver = '{ODBC Driver 17 for SQL Server}'
    connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes;'
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    print(" Connected to Database")
    return conn, cursor




def export_tables_to_csv(output_directory):
    # Create a connection and cursor
    conn, cursor = create_connection()

    # Get the table names from the database
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE' AND table_catalog='MegaMoldInternationalDB'")
    tables = [row.table_name for row in cursor.fetchall()]

    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Loop through each table and save it as a CSV file
    for table in tables:
        query = f"SELECT * FROM {table}"
        df = pd.read_sql(query, conn)
        file_path = os.path.join(output_directory, f"{table}.csv")
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"Table '{table}' has been exported to '{file_path}'")

    # Close the cursor and connection
    cursor.close()
    conn.close()



output_directory = "C:\\Users\\Research\\MegaMold Application\\Deployment_19042023\\Reporting\\"
export_tables_to_csv(output_directory)
