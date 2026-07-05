import sqlite3
conn = sqlite3.connect('users.db')
conn.execute("UPDATE users SET is_admin=1 WHERE username='admin'")
conn.commit()
print('Done - admin flag set')
conn.close()
