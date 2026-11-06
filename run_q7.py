import sqlite3

conn = sqlite3.connect("data/instamart.db")

query = """
WITH daily AS (
    SELECT
        substr(order_ts, 1, 10) AS day,
        SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY substr(order_ts, 1, 10)
)
SELECT
    day,
    revenue,
    SUM(revenue) OVER (
        ORDER BY day
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_revenue
FROM daily
ORDER BY day;
"""

results = conn.execute(query).fetchall()

print("First 10 rows:")
for row in results[:10]:
    print(row)

print("Rows returned:", len(results))

if results:
    print("Final cumulative revenue:", results[-1][-1])

conn.close()