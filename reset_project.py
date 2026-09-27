import sqlite3

DB_NAME = "resume_screening.db"

conn = sqlite3.connect(DB_NAME)
cur = conn.cursor()

print("Starting fresh project reset...")

# Foreign-key checks temporarily off
cur.execute("PRAGMA foreign_keys = OFF")

# Delete testing/user-generated data
tables_to_clear = [
    "applications",
    "candidates",
    "users",
    "jobs",
    "hr_users"
]

for table in tables_to_clear:
    try:
        cur.execute(f"DELETE FROM {table}")
        print(f"Cleared: {table}")
    except sqlite3.OperationalError as e:
        print(f"Skipped {table}: {e}")

# Reset auto-increment counters where available
try:
    cur.execute("DELETE FROM sqlite_sequence")
except sqlite3.OperationalError:
    pass

conn.commit()

# Turn foreign keys back on
cur.execute("PRAGMA foreign_keys = ON")

conn.close()

print()
print("===================================")
print("PROJECT RESET COMPLETE")
print("===================================")
print("Admin, HR, Jobs, Candidates,")
print("Applications and Candidate Users")
print("have been cleared.")
print()
print("Now create fresh HR/Admin data.")